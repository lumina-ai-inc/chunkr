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

// interface AuthorField { // This interface is not strictly needed if authorInfo is directly typed in BlogPostEntry
//   name: string | null;
// }

export interface BlogPostEntry {
  sys: {
    id: string;
  };
  title: string | null;
  subheadings: string | null; // Used as subtitle
  image: ImageField | null;
  publishedDate: string | null;
  slug: string | null;
  authorInfo?: {
    name: string | null;
    // Potentially add other author fields like bio, picture if needed for the blog post page
    picture?: ImageField | null; // Example: if author has a profile picture
  } | null;
  body?: {
    json: import("@contentful/rich-text-types").Document; // Specific type for Rich Text JSON
    links?: {
      assets: {
        block: (ImageField & { sys: { id: string }; contentType?: string })[]; // Typed linked assets
      };
      entries: {
        block: {
          sys: { id: string };
          __typename: string;
          [key: string]: unknown;
        }[]; // Using unknown instead of any
      };
    };
  } | null;
  // SEO fields if you add them in Contentful (e.g., metaDescription, metaTitle)
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
  }[]; // Specific type for locations
}

// Helper function to construct image alt text prioritizing description, then title
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
      blogPageCollection(order: publishedDate_DESC) { # Order by most recent
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
          authorInfo { # This assumes 'authorInfo' is the field ID in Contentful for the author link
                     # And the linked content type has a 'name' field.
                     # If your author content type is named e.g., 'authorProfile', use:
                     # ... on AuthorProfile { name }
            ... on BlogPostAuthor { # Corrected based on the error message
              name
            }
          }
          # Add other fields like tags or readingTime if they exist in your Contentful model
          # tags
          # readingTime
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

    // Log the fetched items to inspect their structure, especially authorInfo
    // console.log('Fetched items:', body.data.blogPageCollection.items);

    return body.data.blogPageCollection.items;
  } catch (error) {
    console.error("Error fetching blog posts from Contentful:", error);
    return [];
  }
}

interface FetchBlogPostBySlugResponse {
  data: {
    blogPageCollection: {
      items: BlogPostEntry[]; // Expecting one item or empty array
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
            width # SEO: good to have for images
            height # SEO: good to have for images
          }
          publishedDate
          slug
          authorInfo {
            ... on BlogPostAuthor {
              name
              # Add other author fields here, e.g.:
              # bio {
              #   json # if bio is rich text
              # }
              # picture {
              #   url
              #   title
              #   description
              #   width
              #   height
              # }
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
                  # Add fields for linked entries within rich text if you have any
                  # For example, if you embed other content types:
                  # ... on YourEmbeddedContentType {
                  #   fieldName
                  # }
                }
              }
            }
          }
          # SEO Fields (if you have them in your Contentful model)
          # seoTitle
          # seoDescription
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
