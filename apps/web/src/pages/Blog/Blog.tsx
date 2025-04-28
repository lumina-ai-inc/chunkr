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
    imageUrl: "/placeholder-image.webp", // Replace with actual image path or URL
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
    imageUrl: "/placeholder-image-2.webp", // Replace with actual image path or URL
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
      style={{ minHeight: "100vh", backgroundColor: "#050609" }} // Match Home background
    >
      {/* Sticky Header */}
      <Flex
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          width: "100%",
          backgroundColor: "rgba(5, 6, 9, 0.8)", // Slightly transparent background like Home scrolled
          backdropFilter: "blur(8px)", // Blur effect like Home scrolled
          boxShadow: "0 1px 0 0 rgba(255, 255, 255, 0.05)", // Add border effect like Home scrolled
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
          padding: "64px 24px", // Add padding like Home sections
        }}
      >
        <Flex
          direction="column"
          style={{ maxWidth: "1024px", width: "100%", gap: "48px" }} // Container for blog content
        >
          {/* Updated Blog Title Section */}
          <Flex
            direction="column"
            align="center"
            gap="16px"
            onMouseEnter={() => handleLottieHover(blogIconLottieRef)}
          >
            {/* Tag */}
            <Flex className="yc-tag" gap="8px" align="center">
              {" "}
              {/* Use yc-tag class and align items */}
              {/* Simple SVG Icon (e.g., document/book) */}
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

            {/* Main Title - Using Text styled like feature-bottom-box-title */}
            <Text
              size="9" // Matches Home.tsx feature-bottom-box-title
              weight="medium" // Matches Home.tsx feature-bottom-box-title
              align="center"
              className="feature-bottom-box-title" // Use class from Home.css
            >
              Blog Posts
            </Text>

            {/* Subtitle - Using Text styled like feature-left-box-subtitle */}
            <Text
              size="5" // Matches Home.tsx feature-left-box-subtitle
              weight="medium" // Matches Home.tsx feature-left-box-subtitle
              className="feature-left-box-subtitle" // Use class from Home.css
              align="center"
              mt="16px" // Add margin like Home.tsx
              style={{ maxWidth: "600px", color: "#ffffffbc" }} // Add max-width and subtle color
            >
              Explore the latest articles, tutorials, and news from our team.
            </Text>
          </Flex>

          {/* Blog Post List */}
          <Flex direction="column" gap="40px">
            {blogPosts.map((post) => (
              <Flex key={post.id} className="blog-card">
                <Flex align="stretch">
                  {/* Image Placeholder */}
                  <Box className="blog-card-image-container">
                    <img
                      src={post.imageUrl}
                      alt={post.imageAlt} // Important for SEO and accessibility
                      className="blog-card-image"
                    />
                  </Box>

                  {/* Text Content */}
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
