import { useState } from 'react';
import { useMutation } from 'react-query';
import { toast } from 'react-hot-toast';
import { Flex, Text, Button, TextArea, Dialog } from '@radix-ui/themes';

interface NotifyWatchersModalProps {
  shareId: string;
  isOpen: boolean;
  onClose: () => void;
}

export function NotifyWatchersModal({
  shareId,
  isOpen,
  onClose,
}: NotifyWatchersModalProps) {
  const [message, setMessage] = useState('');

  const notifyMutation = useMutation({
    mutationFn: async (notificationMessage: string) => {
      const { notifyWatchers } = await import('../../services/liveShareApi');
      await notifyWatchers(shareId, notificationMessage);
    },
    onSuccess: () => {
      toast.success('Watchers notified successfully!');
      setMessage('');
      onClose();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Failed to notify watchers');
    },
  });

  const handleSend = () => {
    if (!message.trim()) {
      toast.error('Please enter a message');
      return;
    }
    notifyMutation.mutate(message.trim());
  };

  const handleClose = () => {
    setMessage('');
    onClose();
  };

  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <Dialog.Content style={{ maxWidth: '540px' }}>
        <Dialog.Title>
          Notify Watchers
        </Dialog.Title>

        <Flex direction="column" gap="16px" style={{ marginTop: '16px' }}>
          <Text size="2" style={{ color: '#666' }}>
            Send an update to all watchers and contacts following this deal.
          </Text>

          <Flex direction="column" gap="8px">
            <Text size="2" weight="medium" style={{ color: '#333' }}>
              Message:
            </Text>
            <TextArea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your update or message for watchers..."
              rows={6}
              style={{ width: '100%', resize: 'vertical' }}
            />
          </Flex>

          <Flex justify="end" gap="12px" style={{ marginTop: '8px' }}>
            <Button
              variant="soft"
              onClick={handleClose}
              style={{ cursor: 'pointer' }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleSend}
              disabled={notifyMutation.isPending || !message.trim()}
              style={{
                cursor: notifyMutation.isPending || !message.trim() ? 'not-allowed' : 'pointer',
                opacity: notifyMutation.isPending || !message.trim() ? 0.5 : 1,
              }}
            >
              {notifyMutation.isPending ? 'Sending...' : 'Send'}
            </Button>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
