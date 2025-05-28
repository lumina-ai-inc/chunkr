const CONTENTFUL_SPACE_ID = import.meta.env.VITE_CONTENTFUL_SPACE_ID;
const CONTENTFUL_ACCESS_TOKEN = import.meta.env.VITE_CONTENTFUL_ACCESS_TOKEN;
const CONTENTFUL_GRAPHQL_ENDPOINT = `https://graphql.contentful.com/content/v1/spaces/${CONTENTFUL_SPACE_ID}/`;

export interface ImageField {
  title: string | null;
  description: string | null;
  url: string | null;
  width?: number | null;
  height?: number | null;
  contentType?: string | null;
}

export interface BlogPostEntry {
  sys: {
    id: string;
  };
  title: string | null;
  subheadings: string | null;
  image: ImageField | null;
  publishedDate: string | null;
  slug: string | null;
  authorsCollection?: {
    items: Array<{
      name: string | null;
      picture?: ImageField | null;
    }>;
  } | null;
  body?: {
    json: import("@contentful/rich-text-types").Document; // Specific type for Rich Text JSON
    links?: {
      assets: {
        block: (ImageField & { sys: { id: string }; contentType?: string })[];
      };
      entries: {
        block: {
          sys: { id: string };
          __typename: string;
          [key: string]: unknown;
        }[];
      };
    };
  } | null;

  seoTitle?: string | null;
  seoDescription?: string | null;
}

interface FetchBlogPostsResponse {
  data: {
    blogPageCollection: {
      items: BlogPostEntry[];
    };
  };
  errors?: {
    message: string;
    locations?: { line: number; column: number }[];
    path?: string[];
  }[];
}

export const getImageAltText = (
  image: ImageField | null | undefined
): string => {
  if (!image) return "";
  return image.description || image.title || "Blog post image";
};

export async function fetchBlogPosts(): Promise<BlogPostEntry[]> {
  if (!CONTENTFUL_SPACE_ID || !CONTENTFUL_ACCESS_TOKEN) {
    console.error(
      "Contentful Space ID or Access Token is not defined. Please check your .env file."
    );
    return [];
  }

  const query = `
    query {
      blogPageCollection(order: publishedDate_DESC, limit: 15) { # Conservative limit
        items {
          sys {
            id
          }
          title
          subheadings
          image {
            title
            description
            url
          }
          publishedDate
          slug
          authorsCollection(limit: 2) { # Limit to 2 authors per post
            items {
              ... on BlogPostAuthor {
                name
                picture: avatar {
                  url
                }
              }
            }
          }
        }
      }
    }
  `;

  try {
    const response = await window.fetch(CONTENTFUL_GRAPHQL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${CONTENTFUL_ACCESS_TOKEN}`,
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      console.error(
        "Contentful API request failed:",
        response.status,
        await response.text()
      );
      return [];
    }

    const body = (await response.json()) as FetchBlogPostsResponse;

    if (body.errors) {
      console.error("Contentful GraphQL Errors:", body.errors);
      return [];
    }

    return body.data.blogPageCollection.items;
  } catch (error) {
    console.error("Error fetching blog posts from Contentful:", error);
    return [];
  }
}

interface FetchBlogPostBySlugResponse {
  data: {
    blogPageCollection: {
      items: BlogPostEntry[];
    };
  };
  errors?: {
    message: string;
    locations?: { line: number; column: number }[];
    path?: string[];
  }[];
}

export async function fetchBlogPostBySlug(
  slug: string
): Promise<BlogPostEntry | null> {
  if (!CONTENTFUL_SPACE_ID || !CONTENTFUL_ACCESS_TOKEN) {
    console.error(
      "Contentful Space ID or Access Token is not defined. Please check your .env file."
    );
    return null;
  }

  const query = `
    query GetBlogPostBySlug($slug: String!) {
      blogPageCollection(where: { slug: $slug }, limit: 1) {
        items {
          sys {
            id
          }
          title
          subheadings
          image {
            title
            description
            url
            width
            height
          }
          publishedDate
          slug
          authorsCollection {
            items {
              ... on BlogPostAuthor {
                name
                picture: avatar {  
                  url
                  title
                  description
                  width
                  height
                }
              }
            }
          }
          body {
            json
            links {
              assets {
                block {
                  sys {
                    id
                  }
                  url
                  title
                  description
                  width
                  height
                  contentType
                }
              }
              entries {
                block {
                  sys {
                    id
                  }
                }
              }
            }
          }
        }
      }
    }
  `;

  try {
    const response = await window.fetch(CONTENTFUL_GRAPHQL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${CONTENTFUL_ACCESS_TOKEN}`,
      },
      body: JSON.stringify({
        query,
        variables: { slug },
      }),
    });

    if (!response.ok) {
      console.error(
        "Contentful API request failed for slug:",
        slug,
        response.status,
        await response.text()
      );
      return null;
    }

    const body = (await response.json()) as FetchBlogPostBySlugResponse;

    if (body.errors) {
      console.error("Contentful GraphQL Errors for slug:", slug, body.errors);
      return null;
    }

    if (body.data.blogPageCollection.items.length > 0) {
      return body.data.blogPageCollection.items[0];
    }

    return null; // No post found for the slug
  } catch (error) {
    console.error(
      `Error fetching blog post by slug ${slug} from Contentful:`,
      error
    );
    return null;
  }
}

export const optimizeContentfulImage = (
  imageUrl: string | null | undefined,
  options: {
    width?: number;
    height?: number;
    quality?: number;
    format?: "webp" | "jpg" | "png" | "avif";
    fit?: "pad" | "fill" | "scale" | "crop" | "thumb";
  } = {}
): string => {
  if (!imageUrl) return "";

  // Default options for high quality
  const {
    width,
    height,
    quality = 85, // High quality default
    format = "webp", // Modern format with good compression
    fit = "fill",
  } = options;

  // Build query parameters
  const params = new URLSearchParams();

  if (width) params.append("w", width.toString());
  if (height) params.append("h", height.toString());
  if (quality) params.append("q", quality.toString());
  if (format) params.append("f", format);
  if (fit && (width || height)) params.append("fit", fit);

  // Add parameters to URL
  const separator = imageUrl.includes("?") ? "&" : "?";
  return `${imageUrl}${separator}${params.toString()}`;
};
