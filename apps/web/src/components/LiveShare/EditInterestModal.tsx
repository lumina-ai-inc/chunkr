import { useState, useEffect } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { addInvestorInterest, deleteInvestorInterest, type InvestorInterest } from '../../services/liveShareApi';
import { toast } from 'react-hot-toast';
import { Flex, Text, Button, TextField, TextArea, RadioGroup, Dialog } from '@radix-ui/themes';

interface EditInterestModalProps {
  shareId: string;
  interest: InvestorInterest | null; // null = add mode, non-null = edit mode
  isOpen: boolean;
  onClose: () => void;
}

type InterestStatus = 'Interested' | 'Maybe' | 'Passed';

export function EditInterestModal({
  shareId,
  interest,
  isOpen,
  onClose,
}: EditInterestModalProps) {
  const isEditMode = interest !== null;
  const [name, setName] = useState('');
  const [status, setStatus] = useState<InterestStatus>('Interested');
  const [amount, setAmount] = useState('');
  const [notes, setNotes] = useState('');
  const queryClient = useQueryClient();

  // Initialize form when interest changes
  useEffect(() => {
    if (interest) {
      setName(interest.name);
      setStatus(interest.status);
      setAmount(interest.amount?.toString() || '');
      setNotes(interest.notes || '');
    } else {
      setName('');
      setStatus('Interested');
      setAmount('');
      setNotes('');
    }
  }, [interest, isOpen]);

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

  const deleteMutation = useMutation({
    mutationFn: () => {
      if (!interest) throw new Error('No interest to delete');
      return deleteInvestorInterest(shareId, interest.id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['interests', shareId]);
      toast.success('Interest deleted!');
      handleClose();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Failed to delete interest');
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!name.trim()) {
      toast.error('Name is required');
      return;
    }

    if (isEditMode && interest) {
      // For edit mode, delete the old one first, then add the updated version
      try {
        await deleteInvestorInterest(shareId, interest.id);
        // Now add the updated version
        await addInvestorInterest(shareId, {
          name,
          status,
          amount: status === 'Passed' ? undefined : parseFloat(amount) || undefined,
          notes: notes || undefined,
        });
        queryClient.invalidateQueries(['interests', shareId]);
        toast.success('Interest updated!');
        handleClose();
      } catch (error: any) {
        toast.error(error.response?.data?.message || 'Failed to update interest');
      }
    } else {
      addMutation.mutate();
    }
  };

  const handleDelete = () => {
    if (window.confirm('Are you sure you want to delete this interest entry?')) {
      deleteMutation.mutate();
    }
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
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <Dialog.Content style={{ maxWidth: '540px' }}>
        <Dialog.Title>
          {isEditMode ? 'Edit Investor Interest' : 'Add Investor Interest'}
        </Dialog.Title>

        <form onSubmit={handleSubmit}>
          <Flex direction="column" gap="16px" mt="16px">
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
            <Flex justify="between" gap="12px" style={{ marginTop: '8px' }}>
              {isEditMode && (
                <Button
                  type="button"
                  variant="soft"
                  color="red"
                  onClick={handleDelete}
                  disabled={deleteMutation.isLoading}
                  style={{ cursor: 'pointer' }}
                >
                  {deleteMutation.isLoading ? 'Deleting...' : 'Delete'}
                </Button>
              )}
              <Flex gap="12px" style={{ marginLeft: 'auto' }}>
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
                  disabled={addMutation.isLoading || deleteMutation.isLoading}
                  style={{
                    cursor: addMutation.isLoading || deleteMutation.isLoading ? 'not-allowed' : 'pointer',
                    opacity: addMutation.isLoading || deleteMutation.isLoading ? 0.5 : 1,
                  }}
                >
                  {addMutation.isLoading || deleteMutation.isLoading ? 'Saving...' : isEditMode ? 'Save' : 'Add'}
                </Button>
              </Flex>
            </Flex>
          </Flex>
        </form>
      </Dialog.Content>
    </Dialog.Root>
  );
}
