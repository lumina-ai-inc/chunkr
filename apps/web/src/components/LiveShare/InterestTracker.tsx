import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  getLiveShare,
  getInvestorInterest,
  deleteLiveShare,
  type LiveShare,
  type InvestorInterest,
} from '../../services/liveShareApi';
import { EditInterestModal } from './EditInterestModal';
import { NotifyWatchersModal } from './NotifyWatchersModal';
import { formatDistanceToNow } from 'date-fns';
import { toast } from 'react-hot-toast';
import { Flex, Text, Button, Badge } from '@radix-ui/themes';

interface InterestTrackerProps {
  shareId: string;
}

export function InterestTracker({ shareId }: InterestTrackerProps) {
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingInterest, setEditingInterest] = useState<InvestorInterest | null>(null);
  const [showNotifyModal, setShowNotifyModal] = useState(false);
  const queryClient = useQueryClient();

  const { data: share } = useQuery<LiveShare>(
    ['liveShare', shareId],
    () => getLiveShare(shareId)
  );

  const { data: interests = [] } = useQuery<InvestorInterest[]>(
    ['interests', shareId],
    () => getInvestorInterest(shareId),
    {
      refetchInterval: 10000,
    }
  );

  const deleteShareMutation = useMutation({
    mutationFn: () => deleteLiveShare(shareId),
    onSuccess: () => {
      queryClient.invalidateQueries(['liveShares']);
      toast.success('Live share deleted successfully');
      // Navigate back or close the view
      window.location.href = '/dashboard';
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Failed to delete live share');
    },
  });

  const handleDeleteShare = () => {
    if (window.confirm('Are you sure you want to delete this live share link? This action cannot be undone.')) {
      deleteShareMutation.mutate();
    }
  };

  const handleCopyLink = () => {
    if (share?.share_url) {
      navigator.clipboard.writeText(share.share_url);
      toast.success('Link copied!');
    }
  };

  const totalAmount = interests
    .filter((i) => i.status !== 'Passed')
    .reduce((sum, i) => sum + (i.amount || 0), 0);

  if (!share) {
    return (
      <Flex p="8px" align="center" justify="center">
        <Text>Loading...</Text>
      </Flex>
    );
  }

  const expiresIn = formatDistanceToNow(new Date(share.expires_at), {
    addSuffix: true,
  });

  return (
    <Flex direction="column" style={{ height: '100%', width: '800px' }}>
      {/* Header */}
      <Flex
        direction="column"
        p="24px"
        style={{
          borderBottom: '1px solid #e0e0e0',
        }}
      >
        <Text size="5" weight="bold" style={{ marginBottom: '1px' }}>
          Live Share Analytics
        </Text>
      </Flex>

      {/* Stats Bar */}
      <Flex
        p="16px 24px"
        align="center"
        justify="between"
        style={{
          borderBottom: '1px solid #e0e0e0',
          backgroundColor: '#f9fafb',
        }}
      >
        <Flex align="center" style={{ flex: 0.75, justifyContent: 'space-evenly' }}>
          <Flex direction="column">
            <Text size="1" style={{ color: '#666', marginBottom: '2px' }}>
              Views
            </Text>
            <Text size="4" weight="bold">
              {share.view_count}
            </Text>
          </Flex>
          <Flex direction="column">
            <Text size="1" style={{ color: '#666', marginBottom: '2px' }}>
              Expires
            </Text>
            <Text size="2" weight="medium">
              {expiresIn}
            </Text>
          </Flex>
          <Flex direction="column">
            <Text size="1" style={{ color: '#666', marginBottom: '2px' }}>
              Status
            </Text>
            <Badge
              color={share.is_expired ? 'red' : 'green'}
              variant="soft"
            >
              {share.is_expired ? 'Expired' : 'Active'}
            </Badge>
          </Flex>
        </Flex>
        <Flex
          onClick={handleCopyLink}
          align="center"
          gap="6px"
          style={{
            cursor: 'pointer',
            padding: '6px 12px',
            borderRadius: '6px',
            backgroundColor: 'transparent',
            transition: 'all 0.2s',
            marginLeft: '16px',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#e0e0e0';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
          title="Copy share link"
        >
          <Text size="3" style={{ color: '#666' }}>
            📋
          </Text>
          <Text size="2" style={{ color: '#666' }}>
            Copy Link
          </Text>
        </Flex>
      </Flex>

      {/* Investor Interest */}
      <Flex
        direction="column"
        style={{ flex: 1, overflowY: 'auto', padding: '28px' }}
      >
        <Flex direction="column" gap="16px">
          <Text size="4" weight="bold">
            Investor Interest
          </Text>
          <Flex align="center" justify="between">
            <Flex
              align="center"
              gap="6px"
              style={{
                cursor: 'pointer',
                padding: '6px 12px',
                borderRadius: '6px',
                backgroundColor: 'transparent',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#f0f0f0';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
              onClick={() => setShowNotifyModal(true)}
            >
              <Text size="3" style={{ color: '#666' }}>
                🔔
              </Text>
              <Text size="2" style={{ color: '#666' }}>
                Notify Watchers
              </Text>
            </Flex>
            <Button
              onClick={() => setShowAddModal(true)}
              style={{ cursor: 'pointer' }}
            >
              + Add
            </Button>
          </Flex>
        </Flex>

        {interests.length === 0 ? (
          <Text size="2" style={{ color: '#666' }}>
            No interest entries yet
          </Text>
        ) : (
          <Flex direction="column" gap="12px">
            {interests.map((interest) => (
              <Flex
                key={interest.id}
                direction="column"
                p="16px"
                onClick={() => setEditingInterest(interest)}
                style={{
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f9fafb';
                  e.currentTarget.style.borderColor = '#1976D2';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.borderColor = '#e0e0e0';
                }}
              >
                <Flex align="start" justify="between" style={{ marginBottom: '8px' }}>
                  <Text size="3" weight="medium">
                    {interest.name}
                  </Text>
                  <Badge
                    color={
                      interest.status === 'Interested'
                        ? 'green'
                        : interest.status === 'Maybe'
                        ? 'yellow'
                        : 'gray'
                    }
                    variant="soft"
                  >
                    {interest.status}
                  </Badge>
                </Flex>

                {interest.amount && (
                  <Text size="4" weight="bold" style={{ marginBottom: '4px' }}>
                    ${interest.amount.toLocaleString()}
                  </Text>
                )}

                {interest.notes && (
                  <Text size="2" style={{ color: '#666', marginTop: '8px' }}>
                    {interest.notes}
                  </Text>
                )}

                <Text size="1" style={{ color: '#999', marginTop: '8px' }}>
                  {formatDistanceToNow(new Date(interest.created_at), {
                    addSuffix: true,
                  })}
                </Text>
              </Flex>
            ))}
          </Flex>
        )}

        {/* Soft Circled Progress */}
        {interests.length > 0 && (
          <Flex
            direction="column"
            p="16px"
            style={{
              marginTop: '24px',
              paddingTop: '16px',
              borderTop: '1px solid #e0e0e0',
            }}
          >
            <Flex align="center" justify="between" style={{ marginBottom: '12px' }}>
              <Text size="2" weight="medium" style={{ color: '#333' }}>
                Soft Circled
              </Text>
              <Text size="4" weight="bold" style={{ color: '#111' }}>
                ${totalAmount.toLocaleString()} / $1.5M Goal
              </Text>
            </Flex>
            {/* Progress Bar */}
            <Flex
              direction="column"
              gap="4px"
            >
              <Flex
                style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: '#e0e0e0',
                  borderRadius: '4px',
                  overflow: 'hidden',
                }}
              >
                <Flex
                  style={{
                    width: `${Math.min((totalAmount / 1500000) * 100, 100)}%`,
                    height: '100%',
                    backgroundColor: '#4CAF50',
                    borderRadius: '4px',
                    transition: 'width 0.3s ease',
                  }}
                />
              </Flex>
              <Text size="1" style={{ color: '#666', textAlign: 'right' }}>
                {Math.round((totalAmount / 1500000) * 100)}% Soft Circled
              </Text>
            </Flex>
          </Flex>
        )}
      </Flex>

      {/* Delete Link Option */}
      <Flex
        p="16px 24px"
        justify="end"
        style={{
          borderTop: '1px solid #e0e0e0',
          backgroundColor: '#f9fafb',
        }}
      >
        <Flex
          onClick={() => !deleteShareMutation.isLoading && handleDeleteShare()}
          align="center"
          gap="6px"
          style={{
            cursor: deleteShareMutation.isLoading ? 'not-allowed' : 'pointer',
            opacity: deleteShareMutation.isLoading ? 0.5 : 1,
            padding: '4px 8px',
            borderRadius: '4px',
            transition: 'background-color 0.2s',
          }}
          onMouseEnter={(e) => {
            if (!deleteShareMutation.isLoading) {
              e.currentTarget.style.backgroundColor = '#fee2e2';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          <Text size="3">🗑️</Text>
          <Text size="2" style={{ color: '#666', fontWeight: '500' }}>
            {deleteShareMutation.isLoading ? 'Deleting...' : 'Delete Link'}
          </Text>
        </Flex>
      </Flex>

      <EditInterestModal
        shareId={shareId}
        interest={null}
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
      />
      <EditInterestModal
        shareId={shareId}
        interest={editingInterest}
        isOpen={!!editingInterest}
        onClose={() => setEditingInterest(null)}
      />
      <NotifyWatchersModal
        shareId={shareId}
        isOpen={showNotifyModal}
        onClose={() => setShowNotifyModal(false)}
      />
    </Flex>
  );
}
