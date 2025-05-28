import { useEffect, useState, useRef, ReactNode } from "react";
import { useParams } from "react-router-dom";
import { Helmet, HelmetProvider } from "react-helmet-async";
import {
  Box,
  Flex,
  Heading,
  Text,
  Avatar,
  Badge,
  ScrollArea,
} from "@radix-ui/themes";
import {
  BLOCKS,
  INLINES,
  MARKS,
  Node,
  Document as ContentfulDocument,
  Text as ContentfulTextNode,
  Block,
  Inline,
} from "@contentful/rich-text-types";
import {
  documentToReactComponents,
  Options,
} from "@contentful/rich-text-react-renderer";
import {
  BlogPostEntry,
  fetchBlogPostBySlug,
  getImageAltText,
  ImageField,
} from "../../services/contentful";
import Header from "../../components/Header/Header";
import Footer from "../../components/Footer/Footer";
import { useAuth } from "react-oidc-context";
import "./BlogPostPage.css";
import Prism from "prismjs";
import BetterButton from "../../components/BetterButton/BetterButton";
import Loader from "../Loader/Loader";

interface HeadingNode {
  id: string;
  text: string;
  level: number;
}

const generateIdFromText = (text: string): string => {
  return text
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "");
};

const getRichTextHeadings = (
  richTextBodyJSON: ContentfulDocument | undefined
): HeadingNode[] => {
  if (!richTextBodyJSON || !richTextBodyJSON.content) {
    return [];
  }
  const headings: HeadingNode[] = [];

  const extractHeadings = (nodes: Node[]) => {
    nodes.forEach((node) => {
      if (node.nodeType && node.nodeType.startsWith("heading-")) {
        const level = parseInt(node.nodeType.split("-")[1], 10);
        const blockNode = node as Block;
        if (
          blockNode.content &&
          blockNode.content.length > 0 &&
          blockNode.content[0].nodeType === BLOCKS.PARAGRAPH
        ) {
          // This case is unlikely for a heading, usually text is direct child
        } else if (
          blockNode.content &&
          blockNode.content.length > 0 &&
          blockNode.content[0].nodeType === "text"
        ) {
          const textNode = blockNode.content[0] as ContentfulTextNode;
          if (textNode.value) {
            const id = generateIdFromText(textNode.value);
            headings.push({ id, text: textNode.value, level });
          }
        }
      }
      const traversableNode = node as Block | Inline;
      if (traversableNode.content && Array.isArray(traversableNode.content)) {
        extractHeadings(traversableNode.content as Node[]);
      }
    });
  };

  extractHeadings(richTextBodyJSON.content);
  return headings.filter((h) => h.level == 2);
};

export default function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>();
  const auth = useAuth();
  const [post, setPost] = useState<BlogPostEntry | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tableOfContents, setTableOfContents] = useState<HeadingNode[]>([]);

  const [isScrolled, setIsScrolled] = useState(false);
  const mainContentRef = useRef<HTMLElement>(null);
  const stickySidebarRef = useRef<HTMLElement>(null);

  // Add scroll offset handler for TOC links
  const handleTocClick = (
    e: React.MouseEvent<HTMLAnchorElement>,
    headingId: string
  ) => {
    e.preventDefault();
    const targetElement = document.getElementById(headingId);
    if (targetElement) {
      const headerOffset = 102; // Same as stickyTopPosition
      const elementPosition = targetElement.getBoundingClientRect().top;
      const offsetPosition =
        elementPosition + window.pageYOffset - headerOffset;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth",
      });
    }
  };

  useEffect(() => {
    const handleScrollAndResize = () => {
      setIsScrolled(window.scrollY > 0);
      if (stickySidebarRef.current && mainContentRef.current) {
        const sidebarElement = stickySidebarRef.current;
        const sidebarParent =
          sidebarElement.parentElement as HTMLElement | null;

        if (!sidebarParent) return;

        const definedSidebarWidth = 300;
        const sidebarInitialTopInParent = 32;
        const stickyTopPosition = 124;

        const sidebarRect = sidebarElement.getBoundingClientRect();
        const parentRect = sidebarParent.getBoundingClientRect();

        if (parentRect.top + sidebarInitialTopInParent > stickyTopPosition) {
          sidebarElement.style.position = "absolute";
          sidebarElement.style.top = `${sidebarInitialTopInParent}px`;
          sidebarElement.style.left = "auto";
          sidebarElement.style.right = "0px";
          sidebarElement.style.width = `${definedSidebarWidth}px`;
          sidebarElement.style.bottom = "auto";
        } else {
          if (stickyTopPosition + sidebarRect.height < parentRect.bottom) {
            sidebarElement.style.position = "fixed";
            sidebarElement.style.top = `${stickyTopPosition}px`;
            const fixedLeft = parentRect.right - definedSidebarWidth;
            sidebarElement.style.left = `${fixedLeft}px`;
            sidebarElement.style.width = `${definedSidebarWidth}px`;
            sidebarElement.style.right = "auto";
            sidebarElement.style.bottom = "auto";
          } else {
            sidebarElement.style.position = "absolute";
            sidebarElement.style.top = "auto";
            sidebarElement.style.bottom = "0px";
            sidebarElement.style.left = "auto";
            sidebarElement.style.right = "0px";
            sidebarElement.style.width = `${definedSidebarWidth}px`;
          }
        }
      }
    };

    window.addEventListener("scroll", handleScrollAndResize);
    window.addEventListener("resize", handleScrollAndResize);
    handleScrollAndResize();
    return () => {
      window.removeEventListener("scroll", handleScrollAndResize);
      window.removeEventListener("resize", handleScrollAndResize);
    };
  }, [post]);

  useEffect(() => {
    if (!slug) {
      setError("No blog post slug provided.");
      setIsLoading(false);
      return;
    }

    const loadPost = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const fetchedPost = await fetchBlogPostBySlug(slug);
        if (fetchedPost) {
          setPost(fetchedPost);
          setTableOfContents(getRichTextHeadings(fetchedPost.body?.json));
        } else {
          setError("Blog post not found.");
        }
      } catch (err) {
        console.error("Error fetching blog post:", err);
        setError("Failed to load the blog post.");
      } finally {
        setIsLoading(false);
      }
    };

    loadPost();
  }, [slug]);

  useEffect(() => {
    if (post) {
      Prism.highlightAll();
    }
  }, [post]);

  const renderOptions: Options = {
    renderMark: {
      [MARKS.BOLD]: (text: ReactNode) => <Text weight="bold">{text}</Text>,
      [MARKS.ITALIC]: (text: ReactNode) => (
        <em style={{ fontStyle: "italic" }}>{text}</em>
      ),
      [MARKS.CODE]: (textNode: ReactNode) => {
        const codeContent =
          typeof textNode === "string" ? textNode : String(textNode);

        return (
          <Badge
            style={{
              backgroundColor: "unset",
              border: "unset",
              maxWidth: "100%",
            }}
          >
            <pre
              style={{
                margin: 0,
                whiteSpace: "pre-wrap",
                padding: "8px !important",
                maxWidth: "100%",
                borderRadius: "4px",
              }}
            >
              <code className="language-python">{codeContent}</code>
            </pre>
          </Badge>
        );
      },
    },
    renderNode: {
      [BLOCKS.HEADING_1]: (node: Node, children: ReactNode) => {
        const blockNode = node as Block;
        const textContent =
          (blockNode.content[0] as ContentfulTextNode)?.value || "";
        return (
          <Heading
            as="h1"
            size="8"
            mt="6"
            mb="4"
            id={generateIdFromText(textContent)}
          >
            {children}
          </Heading>
        );
      },
      [BLOCKS.HEADING_2]: (node: Node, children: ReactNode) => {
        const blockNode = node as Block;
        const textContent =
          (blockNode.content[0] as ContentfulTextNode)?.value || "";
        return (
          <Heading
            as="h2"
            size="7"
            mt="6"
            mb="3"
            id={generateIdFromText(textContent)}
          >
            {children}
          </Heading>
        );
      },
      [BLOCKS.HEADING_3]: (node: Node, children: ReactNode) => {
        const blockNode = node as Block;
        const textContent =
          (blockNode.content[0] as ContentfulTextNode)?.value || "";
        return (
          <Heading
            as="h3"
            size="6"
            mt="5"
            mb="2"
            id={generateIdFromText(textContent)}
          >
            {children}
          </Heading>
        );
      },
      [BLOCKS.HEADING_4]: (_node: Node, children: ReactNode) => (
        <Heading as="h4" size="5" mt="4" mb="2">
          {children}
        </Heading>
      ),
      [BLOCKS.HEADING_5]: (_node: Node, children: ReactNode) => (
        <Heading as="h5" size="4" mt="4" mb="2">
          {children}
        </Heading>
      ),
      [BLOCKS.HEADING_6]: (_node: Node, children: ReactNode) => (
        <Heading as="h6" size="3" mt="4" mb="2">
          {children}
        </Heading>
      ),
      [BLOCKS.PARAGRAPH]: (_node: Node, children: ReactNode) => (
        <Text as="p" size="3" my="3" style={{ lineHeight: "1.7" }}>
          {children}
        </Text>
      ),
      [BLOCKS.UL_LIST]: (_node: Node, children: ReactNode) => (
        <ul style={{ marginLeft: "20px", listStyleType: "disc" }}>
          {children}
        </ul>
      ),
      [BLOCKS.OL_LIST]: (_node: Node, children: ReactNode) => (
        <ol style={{ marginLeft: "20px", listStyleType: "decimal" }}>
          {children}
        </ol>
      ),
      [BLOCKS.LIST_ITEM]: (_node: Node, children: ReactNode) => (
        <li style={{ margin: "8px 0" }}>
          <Text as="span" size="3">
            {children}
          </Text>
        </li>
      ),
      [BLOCKS.QUOTE]: (_node: Node, children: ReactNode) => (
        <Box
          my="4"
          pl="4"
          style={{
            borderLeft: "3px solid var(--accent-9)",
            color: "var(--gray-11)",
          }}
        >
          <Text as="p" style={{ fontStyle: "italic" }}>
            {children}
          </Text>
        </Box>
      ),
      [BLOCKS.HR]: () => (
        <hr style={{ margin: "32px 0", borderColor: "var(--gray-5)" }} />
      ),
      [BLOCKS.EMBEDDED_ASSET]: (node: Node) => {
        const assetNode = node as Block | Inline;
        if (
          !assetNode.data ||
          !assetNode.data.target ||
          !assetNode.data.target.sys ||
          !assetNode.data.target.sys.id
        ) {
          return null;
        }
        const asset = post?.body?.links?.assets?.block?.find(
          (a) => a.sys.id === assetNode.data.target.sys.id
        ) as ImageField | undefined;
        if (asset && asset.url && asset.contentType?.startsWith("image/")) {
          return (
            <Box my="6" className="blog-post-image-container">
              <img
                src={asset.url}
                alt={getImageAltText(asset) || "Embedded blog image"}
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "var(--radius-3)",
                }}
                width={asset.width || undefined}
                height={asset.height || undefined}
                loading="lazy"
              />
              {asset.description && (
                <Text as="p" size="2" color="gray" mt="2" align="center">
                  {asset.description}
                </Text>
              )}
            </Box>
          );
        }
        return null;
      },
      [INLINES.HYPERLINK]: (node: Node, children: ReactNode) => {
        const linkNode = node as Inline;
        if (!linkNode.data || !linkNode.data.uri) return <>{children}</>;
        return (
          <a
            href={linkNode.data.uri as string}
            target="_blank"
            rel="noopener noreferrer"
            className="text-link"
          >
            {children}
          </a>
        );
      },
    },
  };

  const pageTitle = post?.seoTitle || post?.title || "Blog Post";
  const pageDescription =
    post?.seoDescription ||
    post?.subheadings ||
    "Read this interesting blog post.";
  const pageImage = post?.image?.url;

  if (isLoading) {
    return (
      <Flex
        align="center"
        justify="center"
        style={{ minHeight: "100vh", color: "white" }}
      >
        <Loader />
      </Flex>
    );
  }

  if (error) {
    return (
      <Flex
        align="center"
        justify="center"
        style={{ minHeight: "100vh", color: "white" }}
      >
        <Text>Error: {error}</Text>
      </Flex>
    );
  }

  if (!post) {
    return (
      <Flex
        align="center"
        justify="center"
        style={{ minHeight: "100vh", color: "white" }}
      >
        <Text>Blog post not found.</Text>
      </Flex>
    );
  }

  // Extract authors list for multi-author support
  const authors = post.authorsCollection?.items ?? [];

  return (
    <HelmetProvider>
      <Helmet>
        <title>{`${pageTitle} - Chunkr Blog`}</title>
        <meta name="description" content={pageDescription} />
        <meta property="og:title" content={`${pageTitle} - Chunkr Blog`} />
        <meta property="og:description" content={pageDescription} />
        {pageImage && <meta property="og:image" content={pageImage} />}
        <meta property="og:type" content="article" />
        {post.publishedDate && (
          <meta
            property="article:published_time"
            content={new Date(post.publishedDate).toISOString()}
          />
        )}
        {authors.map((author) =>
          author.name ? (
            <meta
              property="article:author"
              content={author.name}
              key={author.name}
            />
          ) : null
        )}
      </Helmet>

      <Flex
        direction="column"
        style={{ minHeight: "100vh", backgroundColor: "#050609" }}
      >
        <Flex
          className={`blog-header-container ${isScrolled ? "scrolled" : ""}`}
          style={{ position: "sticky", top: 0, zIndex: 1001, width: "100%" }}
          justify="center"
        >
          <div
            style={{ maxWidth: "1386px", width: "100%", height: "fit-content" }}
          >
            <Header auth={auth} />
          </div>
        </Flex>

        <Flex
          key={slug}
          direction="column"
          style={{
            maxWidth: "1024px",
            width: "100%",
            margin: "0 auto",
            padding: "32px 24px",
            position: "relative",
          }}
        >
          <header style={{ marginBottom: "32px" }}>
            <Heading
              as="h1"
              size="9"
              weight="bold"
              mb="3"
              className="blog-post-title"
            >
              {post.title}
            </Heading>
            {post.publishedDate && (
              <Box mb="5">
                <Text size="2" color="gray">
                  Published{" "}
                  {new Date(post.publishedDate).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </Text>
              </Box>
            )}
            {post.image?.url && (
              <Box mb="6" className="blog-post-main-image-container">
                <img
                  src={post.image.url}
                  alt={getImageAltText(post.image)}
                  style={{
                    width: "100%",
                    maxHeight: "500px",
                    objectFit: "cover",
                    borderRadius: "var(--radius-4)",
                  }}
                  width={post.image.width || undefined}
                  height={post.image.height || undefined}
                />
              </Box>
            )}
          </header>

          <Flex
            align="start"
            style={{
              width: "100%",
              position: "relative",
            }}
          >
            <section
              ref={mainContentRef as React.RefObject<HTMLElement>}
              className="blog-post-main-content"
              style={{
                flex: "1",
                minWidth: 0,
                marginRight: "calc(300px + 48px)",
              }}
            >
              <article>
                {post.body?.json &&
                  documentToReactComponents(post.body.json, renderOptions)}
              </article>
            </section>

            <section
              ref={stickySidebarRef as React.RefObject<HTMLElement>}
              style={{
                width: "300px",
                position: "absolute",
                top: "32px",
                right: "0px",
                zIndex: 1000,
              }}
              className="blog-post-sidebar"
            >
              <Flex
                direction="column"
                style={{
                  padding: "24px",
                  paddingTop: "0px",
                }}
              >
                {authors.length > 0 && (
                  <Box mb="5">
                    <Heading
                      as="h3"
                      size="4"
                      mb="3"
                      style={{ color: "var(--gray-2)" }}
                    >
                      Written by
                    </Heading>
                    <Flex direction="column" gap="3">
                      {authors.map((author, idx) => (
                        <Flex key={idx} gap="3" align="center">
                          {author.picture?.url ? (
                            <Avatar
                              src={author.picture.url}
                              fallback={author.name?.substring(0, 1) || "A"}
                              alt={author.name || "Author"}
                              size="3"
                              radius="full"
                            />
                          ) : (
                            <Avatar
                              fallback={author.name?.substring(0, 1) || "A"}
                              size="3"
                              radius="full"
                            />
                          )}
                          <Text
                            weight="medium"
                            style={{ color: "var(--gray-2)" }}
                          >
                            {author.name}
                          </Text>
                        </Flex>
                      ))}
                    </Flex>
                  </Box>
                )}

                {tableOfContents.length > 0 && (
                  <Box mb="5">
                    <Heading
                      as="h3"
                      size="4"
                      mb="16px"
                      style={{ color: "var(--gray-2)" }}
                    >
                      Content
                    </Heading>
                    <ScrollArea
                      type="auto"
                      scrollbars="vertical"
                      style={{ maxHeight: "calc(100vh - 200px)" }}
                    >
                      <ul style={{ listStyle: "none", paddingLeft: "0" }}>
                        {tableOfContents.map((heading) => (
                          <li
                            key={heading.id}
                            style={{
                              marginBottom: "12px",
                            }}
                          >
                            <a
                              href={`#${heading.id}`}
                              className="toc-link"
                              onClick={(e) => handleTocClick(e, heading.id)}
                            >
                              <Text
                                size="2"
                                color="gray"
                                style={{
                                  color: "var(--gray-9)",
                                  fontSize: "16px",
                                }}
                              >
                                {heading.text}
                              </Text>
                            </a>
                          </li>
                        ))}
                      </ul>
                    </ScrollArea>
                  </Box>
                )}

                <Flex gap="3">
                  <BetterButton
                    onClick={() => {
                      auth?.signinRedirect({
                        state: { returnTo: "/dashboard" },
                      });
                    }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 30 30"
                      fill="none"
                    >
                      <path
                        d="M7.35 12.225C8.03148 12.4978 8.77803 12.5646 9.4971 12.4171C10.2162 12.2695 10.8761 11.9142 11.3952 11.3952C11.9142 10.8761 12.2695 10.2162 12.4171 9.4971C12.5646 8.77803 12.4978 8.03148 12.225 7.35C13.0179 7.13652 13.7188 6.6687 14.2201 6.01836C14.7214 5.36802 14.9954 4.57111 15 3.75C17.225 3.75 19.4001 4.4098 21.2502 5.64597C23.1002 6.88213 24.5422 8.63914 25.3936 10.6948C26.2451 12.7505 26.4679 15.0125 26.0338 17.1948C25.5998 19.3771 24.5283 21.3816 22.955 22.955C21.3816 24.5283 19.3771 25.5998 17.1948 26.0338C15.0125 26.4679 12.7505 26.2451 10.6948 25.3936C8.63914 24.5422 6.88213 23.1002 5.64597 21.2502C4.4098 19.4001 3.75 17.225 3.75 15C4.57111 14.9954 5.36802 14.7214 6.01836 14.2201C6.6687 13.7188 7.13652 13.0179 7.35 12.225Z"
                        stroke="url(#paint0_linear_236_740)"
                        stroke-width="3"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      ></path>
                      <defs>
                        <linearGradient
                          id="paint0_linear_236_740"
                          x1="15"
                          y1="3.75"
                          x2="15"
                          y2="26.25"
                          gradientUnits="userSpaceOnUse"
                        >
                          <stop stop-color="white"></stop>
                          <stop offset="1" stop-color="#DCE4DD"></stop>
                        </linearGradient>
                      </defs>
                    </svg>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "var(--gray-2)" }}
                    >
                      Try Chunkr
                    </Text>
                  </BetterButton>
                  <BetterButton
                    onClick={() => {
                      window.open("https://github.com/lumina-ai-inc/chunkr");
                    }}
                  >
                    <svg
                      width="18"
                      height="18"
                      viewBox="0 0 18 18"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <g clip-path="url(#clip0_132_8)">
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M8.97318 0C4.01125 0 0 4.04082 0 9.03986C0 13.0359 2.57014 16.4184 6.13561 17.6156C6.58139 17.7056 6.74467 17.4211 6.74467 17.1817C6.74467 16.9722 6.72998 16.2538 6.72998 15.5053C4.23386 16.0442 3.71406 14.4277 3.71406 14.4277C3.31292 13.3801 2.71855 13.1108 2.71855 13.1108C1.90157 12.557 2.77806 12.557 2.77806 12.557C3.68431 12.6169 4.15984 13.4849 4.15984 13.4849C4.96194 14.8618 6.25445 14.4727 6.77443 14.2332C6.84863 13.6495 7.08649 13.2454 7.33904 13.021C5.3482 12.8114 3.25359 12.0332 3.25359 8.56084C3.25359 7.57304 3.60992 6.76488 4.17453 6.13635C4.08545 5.9119 3.77339 4.9838 4.2638 3.74161C4.2638 3.74161 5.02145 3.5021 6.7298 4.66953C7.4612 4.47165 8.21549 4.37099 8.97318 4.37014C9.73084 4.37014 10.5032 4.47502 11.2164 4.66953C12.9249 3.5021 13.6826 3.74161 13.6826 3.74161C14.173 4.9838 13.8607 5.9119 13.7717 6.13635C14.3511 6.76488 14.6928 7.57304 14.6928 8.56084C14.6928 12.0332 12.5982 12.7963 10.5924 13.021C10.9194 13.3053 11.2015 13.844 11.2015 14.6972C11.2015 15.9094 11.1868 16.8823 11.1868 17.1816C11.1868 17.4211 11.3503 17.7056 11.7959 17.6158C15.3613 16.4182 17.9315 13.0359 17.9315 9.03986C17.9462 4.04082 13.9202 0 8.97318 0Z"
                          fill="white"
                        ></path>
                      </g>
                      <defs>
                        <clipPath id="clip0_132_8">
                          <rect width="18" height="17.6327" fill="white"></rect>
                        </clipPath>
                      </defs>
                    </svg>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "var(--gray-2)" }}
                    >
                      Github
                    </Text>
                  </BetterButton>
                </Flex>
              </Flex>
            </section>
          </Flex>
        </Flex>
        <Footer />
      </Flex>
    </HelmetProvider>
  );
}
