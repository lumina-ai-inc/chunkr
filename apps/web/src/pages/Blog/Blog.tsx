import { Flex, Text, Heading, Box } from "@radix-ui/themes";
import Header from "../../components/Header/Header";
import Footer from "../../components/Footer/Footer";
import { useAuth } from "react-oidc-context";
import "./Blog.css"; // Import the CSS file
import Lottie, { LottieRefCurrentProps } from "lottie-react";
import { useRef, useEffect } from "react";
import blogIcon from "../../assets/animations/blog.json";

// Placeholder data - replace with your CMS data later
const blogPosts = [
  {
    id: 1,
    title: "Understanding Modern Web Development",
    subtitle: "A deep dive into the tools and techniques shaping the web.",
    imageUrl: "/placeholder-image.webp",
    imageAlt: "Abstract representation of web development concepts",
    readingTime: "7 min read",
    datePublished: "2024-07-26",
    writer: "Jane Doe",
    tags: ["Web Dev", "React", "TypeScript", "Performance"],
  },
  {
    id: 2,
    title: "Optimizing React Applications",
    subtitle: "Strategies for building faster and more efficient React apps.",
    imageUrl: "/placeholder-image-2.webp",
    imageAlt: "Graph showing performance optimization",
    readingTime: "10 min read",
    datePublished: "2024-07-20",
    writer: "John Smith",
    tags: ["React", "Optimization", "Frontend"],
  },
  // Add more placeholder posts as needed
];

export default function Blog() {
  const auth = useAuth();
  const blogIconLottieRef = useRef<LottieRefCurrentProps>(null);

  useEffect(() => {
    if (blogIconLottieRef.current) {
      blogIconLottieRef.current.pause();
    }
  }, []);

  const handleLottieHover = (ref: React.RefObject<LottieRefCurrentProps>) => {
    if (ref.current) {
      ref.current.goToAndPlay(0);
    }
  };

  return (
    <Flex
      direction="column"
      style={{ minHeight: "100vh", backgroundColor: "#050609" }}
    >
      <Flex
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          width: "100%",
          backgroundColor: "rgba(5, 6, 9, 0.8)",
          backdropFilter: "blur(8px)",
          boxShadow: "0 1px 0 0 rgba(255, 255, 255, 0.05)",
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
      <Flex
        direction="column"
        align="center"
        style={{
          flex: 1,
          width: "100%",
          padding: "64px 24px",
        }}
      >
        <Flex
          direction="column"
          style={{ maxWidth: "1024px", width: "100%", gap: "48px" }}
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

            <Text
              size="9"
              weight="medium"
              align="center"
              className="feature-bottom-box-title"
            >
              Blog Posts
            </Text>

            <Text
              size="5"
              weight="medium"
              className="feature-left-box-subtitle"
              align="center"
              mt="16px"
              style={{ maxWidth: "600px", color: "#ffffffbc" }}
            >
              Explore the latest articles, tutorials, and news from our team.
            </Text>
          </Flex>

          {/* Blog Post List */}
          <Flex direction="column" gap="40px">
            {blogPosts.map((post) => (
              <Flex key={post.id} className="blog-card">
                <Flex align="stretch">
                  <Box className="blog-card-image-container">
                    <img
                      src={post.imageUrl}
                      alt={post.imageAlt}
                      className="blog-card-image"
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
                        {post.title}
                      </Heading>
                      <Text
                        as="p"
                        weight="medium"
                        size="3"
                        className="blog-card-subtitle"
                      >
                        {post.subtitle}
                      </Text>
                    </Flex>

                    <Flex
                      align="center"
                      gap="8px"
                      wrap="wrap"
                      className="blog-card-metadata"
                    >
                      <Text size="1" weight="medium">
                        {post.writer}
                      </Text>
                      <Text size="1" weight="medium">
                        •
                      </Text>
                      <time
                        dateTime={post.datePublished}
                        style={{ display: "flex", alignItems: "center" }}
                      >
                        <Text size="1" weight="medium">
                          {new Date(post.datePublished).toLocaleDateString(
                            "en-US",
                            {
                              year: "numeric",
                              month: "short",
                              day: "numeric",
                            }
                          )}
                        </Text>
                      </time>
                      <Text size="1" weight="medium">
                        •
                      </Text>
                      <Text size="1" weight="medium">
                        {post.readingTime}
                      </Text>
                    </Flex>
                    <Flex gap="16px" wrap="wrap" className="blog-card-tags">
                      {post.tags.map((tag) => (
                        <Flex key={tag} className="blog-card-tag">
                          <Text size="1" weight="medium">
                            {tag}
                          </Text>
                        </Flex>
                      ))}
                    </Flex>
                  </Flex>
                </Flex>
              </Flex>
            ))}
          </Flex>
        </Flex>
      </Flex>

      <Footer />
    </Flex>
  );
}
