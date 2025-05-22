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
// import "prismjs/themes/prism-okaidia.css";

interface HeadingNode {
  id: string;
  text: string;
  level: number; // 1 for H1, 2 for H2, etc.
}

// Helper to generate ID from text
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
  return headings.filter((h) => h.level >= 2 && h.level <= 3); // Only H2 and H3 for TOC
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

  useEffect(() => {
    const handleScrollAndResize = () => {
      setIsScrolled(window.scrollY > 0);
      if (stickySidebarRef.current && mainContentRef.current) {
        const sidebarElement = stickySidebarRef.current;
        const sidebarParent =
          sidebarElement.parentElement as HTMLElement | null;

        if (!sidebarParent) return;

        const definedSidebarWidth = 300; // Define consistent width
        const sidebarInitialTopInParent = 32;
        const stickyTopPosition = 124;

        const sidebarRect = sidebarElement.getBoundingClientRect();
        const parentRect = sidebarParent.getBoundingClientRect();

        if (parentRect.top + sidebarInitialTopInParent > stickyTopPosition) {
          // State A: Initial Absolute (top-aligned in parent) - Not scrolled enough
          sidebarElement.style.position = "absolute";
          sidebarElement.style.top = `${sidebarInitialTopInParent}px`;
          sidebarElement.style.left = "auto";
          sidebarElement.style.right = "0px";
          sidebarElement.style.width = `${definedSidebarWidth}px`; // Use defined width
          sidebarElement.style.bottom = "auto";
        } else {
          // State B or C: Scrolled enough to be either fixed or absolute-bottom.
          // Check if fixing it would make it scroll past the bottom of its parent.
          if (stickyTopPosition + sidebarRect.height < parentRect.bottom) {
            // State B: Fixed to Viewport Top (enough space in parent)
            sidebarElement.style.position = "fixed";
            sidebarElement.style.top = `${stickyTopPosition}px`;
            const fixedLeft = parentRect.right - definedSidebarWidth; // Use defined width for calculation
            sidebarElement.style.left = `${fixedLeft}px`;
            sidebarElement.style.width = `${definedSidebarWidth}px`; // Use defined width
            sidebarElement.style.right = "auto";
            sidebarElement.style.bottom = "auto";
          } else {
            // State C: Absolute, Stuck to Parent Bottom (not enough space if fixed)
            sidebarElement.style.position = "absolute";
            sidebarElement.style.top = "auto";
            sidebarElement.style.bottom = "0px";
            sidebarElement.style.left = "auto";
            sidebarElement.style.right = "0px";
            sidebarElement.style.width = `${definedSidebarWidth}px`; // Use defined width
          }
        }
      }
    };

    window.addEventListener("scroll", handleScrollAndResize);
    window.addEventListener("resize", handleScrollAndResize); // Added resize listener
    handleScrollAndResize(); // Initial call to set position correctly
    return () => {
      window.removeEventListener("scroll", handleScrollAndResize);
      window.removeEventListener("resize", handleScrollAndResize); // Clean up resize listener
    };
  }, [post]); // Dependency array remains [post]

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
        // Directly treat all code as Python
        const codeContent =
          typeof textNode === "string" ? textNode : String(textNode);

        return (
          <Badge
            style={{
              backgroundColor: "unset",
              border: "unset",
            }}
          >
            <pre
              style={{
                margin: 0,
                whiteSpace: "pre-wrap",
                padding: "8px !important",
                maxWidth: "628px",
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
      [BLOCKS.HEADING_4]: (node: Node, children: ReactNode) => (
        <Heading as="h4" size="5" mt="4" mb="2">
          {children}
        </Heading>
      ),
      [BLOCKS.HEADING_5]: (node: Node, children: ReactNode) => (
        <Heading as="h5" size="4" mt="4" mb="2">
          {children}
        </Heading>
      ),
      [BLOCKS.HEADING_6]: (node: Node, children: ReactNode) => (
        <Heading as="h6" size="3" mt="4" mb="2">
          {children}
        </Heading>
      ),
      [BLOCKS.PARAGRAPH]: (node: Node, children: ReactNode) => (
        <Text as="p" size="3" my="3" style={{ lineHeight: "1.7" }}>
          {children}
        </Text>
      ),
      [BLOCKS.UL_LIST]: (node: Node, children: ReactNode) => (
        <ul style={{ marginLeft: "20px", listStyleType: "disc" }}>
          {children}
        </ul>
      ),
      [BLOCKS.OL_LIST]: (node: Node, children: ReactNode) => (
        <ol style={{ marginLeft: "20px", listStyleType: "decimal" }}>
          {children}
        </ol>
      ),
      [BLOCKS.LIST_ITEM]: (node: Node, children: ReactNode) => (
        <li style={{ margin: "8px 0" }}>
          <Text as="span" size="3">
            {children}
          </Text>
        </li>
      ),
      [BLOCKS.QUOTE]: (node: Node, children: ReactNode) => (
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
        <Text>Loading post...</Text>
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
        {post.authorInfo?.name && (
          <meta property="article:author" content={post.authorInfo.name} />
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
                  Published on{" "}
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
                {post.authorInfo && (
                  <Box mb="5">
                    <Heading
                      as="h3"
                      size="4"
                      mb="3"
                      style={{ color: "var(--gray-2)" }}
                    >
                      Written by
                    </Heading>
                    <Flex gap="3" align="center">
                      {post.authorInfo.picture?.url ? (
                        <Avatar
                          src={post.authorInfo.picture.url}
                          fallback={
                            post.authorInfo.name?.substring(0, 1) || "A"
                          }
                          alt={post.authorInfo.name || "Author"}
                          size="3"
                          radius="full"
                        />
                      ) : (
                        <Avatar
                          fallback={
                            post.authorInfo.name?.substring(0, 1) || "A"
                          }
                          size="3"
                          radius="full"
                        />
                      )}
                      <Text weight="medium" style={{ color: "var(--gray-2)" }}>
                        {post.authorInfo.name}
                      </Text>
                    </Flex>
                  </Box>
                )}

                {tableOfContents.length > 0 && (
                  <Box>
                    <Heading
                      as="h3"
                      size="4"
                      mb="3"
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
                              marginBottom: "8px",
                            }}
                          >
                            <a href={`#${heading.id}`} className="toc-link">
                              <Text
                                size="2"
                                color="gray"
                                style={{
                                  color:
                                    heading.level === 3
                                      ? "var(--gray-9)"
                                      : "var(--gray-7)",
                                  fontSize:
                                    heading.level === 3 ? "14px" : "16px",
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
              </Flex>
            </section>
          </Flex>
        </Flex>
        <Footer />
      </Flex>
    </HelmetProvider>
  );
}
