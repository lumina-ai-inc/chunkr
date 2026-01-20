import { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from 'react-query';
import { getPublicShare, trackView } from '../../services/liveShareApi';
import { Flex, Text, Button } from '@radix-ui/themes';

export function PublicSharePage() {
  const { shareId } = useParams<{ shareId: string }>();

  const { data, isLoading, error } = useQuery(
    ['publicShare', shareId],
    () => getPublicShare(shareId!),
    {
      enabled: !!shareId,
      retry: false,
    }
  );

  // Track view on mount
  useEffect(() => {
    if (shareId) {
      trackView(shareId, undefined, navigator.userAgent).catch(console.error);
    }
  }, [shareId]);

  if (isLoading) {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            width: '48px',
            height: '48px',
            border: '2px solid #1976D2',
            borderTopColor: 'transparent',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
          }}
        />
      </div>
    );
  }

  if (error) {
    const errorMessage =
      (error as any)?.response?.status === 410
        ? 'This share link has expired.'
        : 'Share link not found or invalid.';

    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#f9fafb',
        }}
      >
        <Flex
          direction="column"
          align="center"
          gap="16px"
          style={{
            maxWidth: '800px',
            width: '100%',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            padding: '32px',
            textAlign: 'center',
          }}
        >
          <Text size="6" weight="bold" style={{ color: '#111' }}>
            {errorMessage}
          </Text>
          <Text size="2" style={{ color: '#666' }}>
            Please contact the deal owner for a new link.
          </Text>
          <Link to="/">
            <Button style={{ cursor: 'pointer' }}>Go to ReFlow</Button>
          </Link>
        </Flex>
      </div>
    );
  }

  const { share, deal } = data!;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb' }}>
      {/* Header */}
      <header
        style={{
          backgroundColor: 'white',
          borderBottom: '1px solid #e0e0e0',
        }}
      >
        <div
          style={{
            maxWidth: '896px',
            margin: '0 auto',
            padding: '16px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Text size="5" weight="bold" style={{ color: '#1976D2' }}>
            ReFlow
          </Text>
          <Text size="2" style={{ color: '#666' }}>
            {share.view_count} {share.view_count === 1 ? 'view' : 'views'}
          </Text>
        </div>
      </header>

      {/* Content */}
      <main
        style={{
          maxWidth: '896px',
          margin: '0 auto',
          padding: '32px 24px',
        }}
      >
        {/* Deal Header */}
        <Flex
          direction="column"
          p="24px"
          style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            marginBottom: '24px',
          }}
        >
          <Text size="7" weight="bold" style={{ color: '#111', marginBottom: '8px' }}>
            {deal.property_name || deal.deal_name || 'Deal Details'}
          </Text>
          {deal.address && (
            <Text size="2" style={{ color: '#666' }}>
              {deal.address}
            </Text>
          )}
        </Flex>

        {/* Quick Stats */}
        {(deal.purchase_price || deal.noi) && (
          <Flex
            direction="column"
            p="24px"
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              marginBottom: '24px',
            }}
          >
            <Text size="4" weight="bold" style={{ marginBottom: '16px' }}>
              Quick Stats
            </Text>
            <Flex gap="16px">
              {deal.purchase_price && (
                <Flex direction="column">
                  <Text size="1" style={{ color: '#666', marginBottom: '4px' }}>
                    Purchase Price
                  </Text>
                  <Text size="5" weight="bold">
                    ${deal.purchase_price.toLocaleString()}
                  </Text>
                </Flex>
              )}
              {deal.noi && (
                <Flex direction="column">
                  <Text size="1" style={{ color: '#666', marginBottom: '4px' }}>
                    NOI
                  </Text>
                  <Text size="5" weight="bold">
                    ${deal.noi.toLocaleString()}
                  </Text>
                </Flex>
              )}
            </Flex>
          </Flex>
        )}

        {/* Deal Summary */}
        {deal.summary && (
          <Flex
            direction="column"
            p="24px"
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              marginBottom: '24px',
            }}
          >
            <Text size="4" weight="bold" style={{ marginBottom: '16px' }}>
              Summary
            </Text>
            <Text size="2" style={{ color: '#333', whiteSpace: 'pre-wrap' }}>
              {deal.summary}
            </Text>
          </Flex>
        )}

        {/* Documents */}
        {deal.documents && deal.documents.length > 0 && (
          <Flex
            direction="column"
            p="24px"
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              marginBottom: '24px',
            }}
          >
            <Text size="4" weight="bold" style={{ marginBottom: '16px' }}>
              Documents
            </Text>
            <Flex direction="column" gap="8px">
              {deal.documents.map((doc: any) => (
                <Flex
                  key={doc.id}
                  align="center"
                  justify="between"
                  p="12px"
                  style={{
                    backgroundColor: '#f9fafb',
                    borderRadius: '6px',
                  }}
                >
                  <Text size="2" style={{ color: '#333' }}>
                    {doc.file_name || doc.fileName}
                  </Text>
                  <Text size="1" style={{ color: '#666' }}>
                    {doc.page_count || doc.pageCount || 0} pages
                  </Text>
                </Flex>
              ))}
            </Flex>
          </Flex>
        )}
      </main>

      {/* Fixed Footer CTA */}
      <footer
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          backgroundColor: 'white',
          borderTop: '1px solid #e0e0e0',
          boxShadow: '0 -4px 6px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div
          style={{
            maxWidth: '896px',
            margin: '0 auto',
            padding: '16px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Text size="2" style={{ color: '#666' }}>
            Want to create your own deal packages?
          </Text>
          <Link to="/signup">
            <Button style={{ cursor: 'pointer' }}>Sign Up Free</Button>
          </Link>
        </div>
      </footer>
    </div>
  );
}
