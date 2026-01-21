import { useState } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { addInvestorInterest } from '../../services/liveShareApi';
import { toast } from 'react-hot-toast';
import { Flex, Text, Button, TextField, TextArea, RadioGroup } from '@radix-ui/themes';

interface AddInterestModalProps {
  shareId: string;
  isOpen: boolean;
  onClose: () => void;
}

type InterestStatus = 'Interested' | 'Maybe' | 'Passed';

export function AddInterestModal({
  shareId,
  isOpen,
  onClose,
}: AddInterestModalProps) {
  const [name, setName] = useState('');
  const [status, setStatus] = useState<InterestStatus>('Interested');
  const [amount, setAmount] = useState('');
  const [notes, setNotes] = useState('');
  const queryClient = useQueryClient();

  const addMutation = useMutation({
    mutationFn: () =>
      addInvestorInterest(shareId, {
        name,
        status,
        amount: status === 'Passed' ? undefined : parseFloat(amount) || undefined,
        notes: notes || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries(['interests', shareId]);
      toast.success('Interest added!');
      handleClose();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Failed to add interest');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!name.trim()) {
      toast.error('Name is required');
      return;
    }

    addMutation.mutate();
  };

  const handleClose = () => {
    setName('');
    setStatus('Interested');
    setAmount('');
    setNotes('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50,
      }}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
          maxWidth: '448px',
          width: '100%',
          padding: '24px',
        }}
      >
        <Text size="5" weight="bold" style={{ marginBottom: '16px', display: 'block' }}>
          Add Investor Interest
        </Text>

        <form onSubmit={handleSubmit}>
          <Flex direction="column" gap="16px">
            {/* Name */}
            <Flex direction="column" gap="4px">
              <Text size="2" weight="medium" style={{ color: '#333' }}>
                Name *
              </Text>
              <TextField.Root
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Investor name"
                required
              />
            </Flex>

            {/* Status */}
            <Flex direction="column" gap="8px">
              <Text size="2" weight="medium" style={{ color: '#333' }}>
                Status *
              </Text>
              <RadioGroup.Root
                value={status}
                onValueChange={(value) => setStatus(value as InterestStatus)}
              >
                <Flex direction="column" gap="8px">
                  {(['Interested', 'Maybe', 'Passed'] as InterestStatus[]).map((s) => (
                    <label
                      key={s}
                      style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
                    >
                      <RadioGroup.Item value={s} />
                      <Text size="2" style={{ color: '#333' }}>{s}</Text>
                    </label>
                  ))}
                </Flex>
              </RadioGroup.Root>
            </Flex>

            {/* Amount */}
            <Flex direction="column" gap="4px">
              <Text size="2" weight="medium" style={{ color: '#333' }}>
                Amount (optional)
              </Text>
              <TextField.Root
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="Investment amount"
                min="0"
                step="1000"
              />
            </Flex>

            {/* Notes */}
            <Flex direction="column" gap="4px">
              <Text size="2" weight="medium" style={{ color: '#333' }}>
                Notes (optional)
              </Text>
              <TextArea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Additional notes"
                rows={3}
              />
            </Flex>

            {/* Actions */}
            <Flex justify="end" gap="12px" style={{ marginTop: '8px' }}>
              <Button
                type="button"
                variant="soft"
                onClick={handleClose}
                style={{ cursor: 'pointer' }}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={addMutation.isPending}
                style={{
                  cursor: addMutation.isPending ? 'not-allowed' : 'pointer',
                  opacity: addMutation.isPending ? 0.5 : 1,
                }}
              >
                {addMutation.isPending ? 'Saving...' : 'Save'}
              </Button>
            </Flex>
          </Flex>
        </form>
      </div>
    </div>
  );
}
