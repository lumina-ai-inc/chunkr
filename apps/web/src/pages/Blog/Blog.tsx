import { Flex, Text, Heading, Box } from "@radix-ui/themes";
import Header from "../../components/Header/Header";
import Footer from "../../components/Footer/Footer";
import { useAuth } from "react-oidc-context";
import "./Blog.css";
import Lottie, { LottieRefCurrentProps } from "lottie-react";
import { useRef, useEffect, useState } from "react";
import blogIcon from "../../assets/animations/blog.json";
import { Link } from "react-router-dom";
import {
  fetchBlogPosts,
  BlogPostEntry,
  getImageAltText,
} from "../../services/contentful";
import Loader from "../Loader/Loader";

export default function Blog() {
  const auth = useAuth();
  const blogIconLottieRef = useRef<LottieRefCurrentProps>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  const [posts, setPosts] = useState<BlogPostEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (blogIconLottieRef.current) {
      blogIconLottieRef.current.pause();
    }

    const loadPosts = async () => {
      setIsLoading(true);
      try {
        const fetchedPosts = await fetchBlogPosts();
        setPosts(fetchedPosts);
      } catch (error) {
        console.error("Failed to fetch blog posts:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadPosts();
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleLottieHover = (ref: React.RefObject<LottieRefCurrentProps>) => {
    if (ref.current) {
      ref.current.goToAndPlay(0);
    }
  };

  const renderBlogPostCard = (post: BlogPostEntry) => {
    const title = post.title || "Untitled Post";
    const subtitle = post.subheadings || "No subtitle available.";
    const imageUrl = post.image?.url || "/placeholder-image.webp";
    const imageAlt = getImageAltText(post.image) || "Blog post image";
    const datePublished = post.publishedDate
      ? new Date(post.publishedDate)
      : new Date();
    const authors = post.authorsCollection?.items ?? [];

    return (
      <Link
        to={`/blog/${post.slug}`}
        key={post.sys.id}
        style={{
          textDecoration: "none",
          color: "inherit",
          flexGrow: 1,
          flexBasis: "0%",
          minWidth: 0,
        }}
      >
        <Flex
          className="blog-card"
          direction="column"
          style={{ height: "100%" }}
        >
          <Flex align="stretch" direction="column">
            <Box className="blog-card-image-container">
              <img
                src={imageUrl}
                alt={imageAlt}
                className="blog-card-image"
                loading="lazy"
              />
            </Box>

            <Flex
              direction="column"
              gap="16px"
              style={{ flex: 1, padding: "32px" }}
            >
              <Flex direction="column" gap="8px">
                <Heading
                  as="h2"
                  size="6"
                  weight="bold"
                  className="blog-card-title"
                >
                  {title}
                </Heading>
                <Text
                  as="p"
                  weight="medium"
                  size="3"
                  className="blog-card-subtitle"
                >
                  {subtitle}
                </Text>
              </Flex>

              <Flex
                direction="column"
                gap="8px"
                wrap="wrap"
                className="blog-card-metadata"
              >
                {authors.length > 0 && (
                  <Text size="1" weight="medium">
                    By{" "}
                    {authors
                      .map((a) => a.name)
                      .filter(Boolean)
                      .join(", ")}
                  </Text>
                )}
                <time
                  dateTime={datePublished.toISOString()}
                  style={{ display: "flex", alignItems: "center" }}
                >
                  <Text size="1" weight="medium">
                    {datePublished.toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </Text>
                </time>
              </Flex>
            </Flex>
          </Flex>
        </Flex>
      </Link>
    );
  };

  return (
    <Flex
      direction="column"
      style={{ minHeight: "100vh", backgroundColor: "#050609" }}
    >
      <Flex className="blog-image-container"></Flex>
      <Flex className="blog-image-container-overlay"></Flex>
      <Flex
        className={`blog-header-container ${isScrolled ? "scrolled" : ""}`}
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          width: "100%",
        }}
        justify="center"
      >
        <div
          style={{
            maxWidth: "1386px",
            width: "100%",
            height: "fit-content",
          }}
        >
          <Header auth={auth} />
        </div>
      </Flex>

      {/* Main Content Area */}
      <Flex className="blog-page-container" direction="column" align="center">
        <Flex
          direction="column"
          style={{ maxWidth: "1386px", width: "100%", gap: "48px" }}
          px="24px"
          py="12px"
        >
          <Flex
            direction="column"
            align="center"
            gap="16px"
            onMouseEnter={() => handleLottieHover(blogIconLottieRef)}
          >
            <Flex className="yc-tag" gap="8px" align="center">
              {" "}
              <Lottie
                lottieRef={blogIconLottieRef}
                animationData={blogIcon}
                loop={false}
                autoplay={false}
                style={{ width: "16px", height: "16px" }}
              />
              <Text
                size="2"
                weight="medium"
                style={{
                  color: "#ffffff",
                  textShadow: "0 0 10px rgba(255, 255, 255, 0.45)",
                  letterSpacing: "0.02em",
                }}
              >
                Insights & Updates
              </Text>
            </Flex>

            <Heading
              as="h1"
              weight="bold"
              align="center"
              className="blog-page-title"
            >
              {" "}
              Chunkr Blog
            </Heading>

            <Text
              size="2"
              weight="medium"
              className="blog-page-description"
              align="center"
              mt="40px"
            >
              Explore the latest research, cookbooks, and updates from our team
            </Text>
          </Flex>

          {/* Blog Post List */}
          <Flex direction="column" gap="40px" style={{ width: "100%" }}>
            {isLoading ? (
              <Loader />
            ) : posts.length === 0 ? (
              <Text>No blog posts found.</Text>
            ) : (
              Array.from({ length: Math.ceil(posts.length / 2) }).map(
                (_, rowIndex) => {
                  const post1Index = rowIndex * 2;
                  const post2Index = rowIndex * 2 + 1;
                  const post1Data = posts[post1Index];
                  const post2Data =
                    posts.length > post2Index ? posts[post2Index] : null;

                  if (!post1Data) {
                    return null;
                  }

                  return (
                    <Flex
                      key={`row-${rowIndex}`}
                      direction={{ initial: "column", sm: "row" }}
                      gap="40px"
                      align="stretch"
                      style={{ width: "100%" }}
                    >
                      {renderBlogPostCard(post1Data)}
                      {post2Data && renderBlogPostCard(post2Data)}
                    </Flex>
                  );
                }
              )
            )}
          </Flex>
        </Flex>
      </Flex>

      <Footer />
    </Flex>
  );
}
