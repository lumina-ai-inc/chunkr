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
  Card,
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
import "./BlogPostPage.css"; // We'll create this for custom styles

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
  const mainContentRef = useRef<HTMLDivElement>(null);
  const stickySidebarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
      if (stickySidebarRef.current && mainContentRef.current) {
        const headerElement = document.querySelector(
          ".blog-header-container"
        ) as HTMLElement;
        const headerHeight = headerElement ? headerElement.offsetHeight : 80;
        const mainContentTop =
          mainContentRef.current.getBoundingClientRect().top + window.scrollY;
        const scrollTop = window.scrollY;

        const sidebarDefaultTop = 32; // Corresponds to parent padding + desired gap

        if (scrollTop > mainContentTop - headerHeight - sidebarDefaultTop) {
          stickySidebarRef.current.style.position = "fixed";
          stickySidebarRef.current.style.top = `${headerHeight + 24}px`;
        } else {
          stickySidebarRef.current.style.position = "absolute";
          stickySidebarRef.current.style.top = `${sidebarDefaultTop}px`;
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
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

  const renderOptions: Options = {
    renderMark: {
      [MARKS.BOLD]: (text: ReactNode) => <Text weight="bold">{text}</Text>,
      [MARKS.ITALIC]: (text: ReactNode) => (
        <em style={{ fontStyle: "italic" }}>{text}</em>
      ),
      [MARKS.CODE]: (text: ReactNode) => (
        <Badge color="gray" variant="soft" highContrast>
          <pre
            style={{
              margin: 0,
              whiteSpace: "pre-wrap",
              fontFamily: "monospace",
            }}
          >
            {text}
          </pre>
        </Badge>
      ),
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
          style={{
            maxWidth: "1386px",
            width: "100%",
            margin: "0 auto",
            padding: "32px 24px",
            gap: "48px",
            position: "relative",
          }}
          align="start"
        >
          <Box style={{ flex: "3", minWidth: 0 }} ref={mainContentRef}>
            {" "}
            {/* Equivalent to 66-75% */}
            <article>
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

              {post.body?.json &&
                documentToReactComponents(post.body.json, renderOptions)}
            </article>
          </Box>

          <aside
            ref={stickySidebarRef as React.RefObject<HTMLElement>}
            style={{
              flex: "1",
              width: "300px",
              maxWidth: "33%",
              position: "absolute",
              right: "24px",
              top: "32px",
              zIndex: 1000,
            }}
            className="blog-post-sidebar"
          >
            <Card variant="surface" style={{ padding: "24px" }}>
              {post.authorInfo && (
                <Box mb="5">
                  <Heading as="h3" size="4" mb="3">
                    Written by
                  </Heading>
                  <Flex gap="3" align="center">
                    {post.authorInfo.picture?.url ? (
                      <Avatar
                        src={post.authorInfo.picture.url}
                        fallback={post.authorInfo.name?.substring(0, 1) || "A"}
                        alt={post.authorInfo.name || "Author"}
                        size="3"
                        radius="full"
                      />
                    ) : (
                      <Avatar
                        fallback={post.authorInfo.name?.substring(0, 1) || "A"}
                        size="3"
                        radius="full"
                      />
                    )}
                    <Text weight="medium">{post.authorInfo.name}</Text>
                  </Flex>
                </Box>
              )}

              {tableOfContents.length > 0 && (
                <Box>
                  <Heading as="h3" size="4" mb="3">
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
                            paddingLeft: heading.level === 3 ? "16px" : "0",
                          }}
                        >
                          <a href={`#${heading.id}`} className="toc-link">
                            <Text size="2" color="gray">
                              {heading.text}
                            </Text>
                          </a>
                        </li>
                      ))}
                    </ul>
                  </ScrollArea>
                </Box>
              )}
            </Card>
          </aside>
        </Flex>
        <Footer />
      </Flex>
    </HelmetProvider>
  );
}
