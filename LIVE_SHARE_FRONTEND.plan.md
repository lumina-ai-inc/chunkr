---
name: Live Share Feature - Frontend & UX
overview: Implement frontend UI components, API integration, routing, and user experience for the live share feature. This includes sidebar updates, modals, public share page, and interest tracking interface.
todos:
  - id: frontend_api
    content: Create frontend API service for live share operations
    status: pending
  - id: sidebar_live_links
    content: "Update LeftNavPane: Remove contacts/shared packages, add LIVE LINKS section with view counts"
    status: pending
    dependencies:
      - frontend_api
  - id: memo_tab_updates
    content: "Update Memo tab: Remove subtitle, rename button to 'Create Live Share', move buttons to top"
    status: pending
  - id: create_share_modal
    content: Create CreateLiveShareModal component with expiration options
    status: pending
    dependencies:
      - frontend_api
  - id: interest_tracker
    content: Create InterestTracker component for right pane with stats and interest list
    status: pending
    dependencies:
      - frontend_api
  - id: add_interest_modal
    content: Create AddInterestModal component for adding investor interest entries
    status: pending
    dependencies:
      - interest_tracker
  - id: public_share_page
    content: Create PublicSharePage component with deal content, quick stats, and view tracking
    status: pending
    dependencies:
      - frontend_api
  - id: routing_integration
    content: Add /share/:shareId route to main router (no auth required)
    status: pending
    dependencies:
      - public_share_page
  - id: right_pane_integration
    content: Update RightPreviewPane to show InterestTracker when live share is selected
    status: pending
    dependencies:
      - interest_tracker
  - id: dashboard_state
    content: Update DashboardThreePane to manage live share selection state
    status: pending
    dependencies:
      - sidebar_live_links
      - right_pane_integration
  - id: cleanup
    content: Remove ShareWithContactsModal and unused contact-related code
    status: pending
    dependencies:
      - memo_tab_updates
---

# Live Share Feature - Frontend & UX Implementation

## Overview

This plan covers all frontend and UX changes for the live share feature, including components, state management, routing, and user interactions.

## Architecture

```
Frontend API Layer (liveShareApi.ts)
    ↓
Dashboard State Management (DashboardThreePane)
    ├─→ Sidebar (LeftNavPane - LIVE LINKS)
    ├─→ Right Pane (InterestTracker)
    └─→ Modals (CreateLiveShareModal, AddInterestModal)

Public Route (no auth)
    └─→ PublicSharePage
```

## Implementation Steps

### 1. Frontend API Service

**File**: `apps/web/src/services/liveShareApi.ts` (new file)

```typescript
import axios from './axios';

export interface LiveShare {
  id: string;
  deal_id: string;
  short_id: string;
  share_url: string;
  expires_at: string;
  view_count: number;
  created_at: string;
  is_expired: boolean;
}

export interface ShareView {
  id: string;
  live_share_id: string;
  viewed_at: string;
  ip_address?: string;
  user_agent?: string;
}

export interface InvestorInterest {
  id: string;
  live_share_id: string;
  name: string;
  amount?: number;
  status: 'Interested' | 'Maybe' | 'Passed';
  notes?: string;
  created_at: string;
}

export interface CreateLiveShareRequest {
  deal_id: string;
  expires_in_days: number;
}

export interface CreateInterestRequest {
  name: string;
  amount?: number;
  status: 'Interested' | 'Maybe' | 'Passed';
  notes?: string;
}

// ============================================================================
// AUTHENTICATED ENDPOINTS
// ============================================================================

export const createLiveShare = async (
  dealId: string,
  expiresInDays: number
): Promise<LiveShare> => {
  const response = await axios.post('/api/v1/live-shares', {
    deal_id: dealId,
    expires_in_days: expiresInDays,
  });
  return response.data;
};

export const getLiveShares = async (): Promise<LiveShare[]> => {
  const response = await axios.get('/api/v1/live-shares');
  return response.data;
};

export const getLiveShare = async (shareId: string): Promise<LiveShare> => {
  const response = await axios.get(`/api/v1/live-shares/${shareId}`);
  return response.data;
};

export const getShareViews = async (shareId: string): Promise<ShareView[]> => {
  const response = await axios.get(`/api/v1/live-shares/${shareId}/views`);
  return response.data;
};

export const addInvestorInterest = async (
  shareId: string,
  interest: CreateInterestRequest
): Promise<InvestorInterest> => {
  const response = await axios.post(
    `/api/v1/live-shares/${shareId}/interest`,
    interest
  );
  return response.data;
};

export const getInvestorInterest = async (
  shareId: string
): Promise<InvestorInterest[]> => {
  const response = await axios.get(`/api/v1/live-shares/${shareId}/interest`);
  return response.data;
};

export const deleteInvestorInterest = async (
  shareId: string,
  interestId: string
): Promise<void> => {
  await axios.delete(`/api/v1/live-shares/${shareId}/interest/${interestId}`);
};

// ============================================================================
// PUBLIC ENDPOINTS (no auth required)
// ============================================================================

export const getPublicShare = async (
  shortId: string
): Promise<{ share: LiveShare; deal: any }> => {
  const response = await axios.get(`/api/v1/public/share/${shortId}`);
  return response.data;
};

export const trackView = async (
  shortId: string,
  ipAddress?: string,
  userAgent?: string
): Promise<void> => {
  await axios.post(`/api/v1/public/share/${shortId}/view`, {
    ip_address: ipAddress,
    user_agent: userAgent,
  });
};
```

---

### 2. Sidebar Updates - LIVE LINKS Section

**File**: `apps/web/src/components/Dashboard/LeftNavPane.tsx`

**Remove** (around lines 150-250):
- Entire "MY CONTACTS" section
- Contact-related imports (`ContactsApi`, `Contact` type)
- Contact state variables

**Add** at the end of the sidebar (after DOCUMENTS section):

```typescript
import { useQuery } from '@tanstack/react-query';
import { getLiveShares } from '../../services/liveShareApi';
import type { LiveShare } from '../../services/liveShareApi';

// Inside component:
const { data: liveShares = [] } = useQuery({
  queryKey: ['liveShares'],
  queryFn: getLiveShares,
  refetchInterval: 30000, // Refresh every 30 seconds
});

// Add prop to component interface:
interface LeftNavPaneProps {
  // ... existing props
  onSelectLiveShare?: (shareId: string) => void;
  selectedLiveShareId?: string | null;
}

// In JSX, after DOCUMENTS section:
<div className="space-y-2">
  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-3 py-2">
    Live Links
  </h3>
  
  {liveShares.length === 0 ? (
    <p className="text-xs text-gray-500 px-3 py-2">
      No active share links yet
    </p>
  ) : (
    <div className="space-y-1">
      {liveShares.map((share) => {
        const deal = deals.find((d) => d.id === share.deal_id);
        const isExpired = share.is_expired;
        const isSelected = selectedLiveShareId === share.id;
        
        return (
          <button
            key={share.id}
            onClick={() => !isExpired && onSelectLiveShare?.(share.id)}
            className={`
              w-full text-left px-3 py-2 rounded-md text-sm
              flex items-center justify-between gap-2
              transition-colors
              ${isSelected
                ? 'bg-blue-50 text-blue-700 font-medium'
                : isExpired
                ? 'text-gray-400 cursor-not-allowed'
                : 'text-gray-700 hover:bg-gray-100'
              }
            `}
            disabled={isExpired}
          >
            <div className="flex items-center gap-2 min-w-0">
              {!isExpired && (
                <span className="text-green-500 text-xs">●</span>
              )}
              <span className="truncate">
                {deal?.property_name || 'Unknown Deal'}
              </span>
            </div>
            <span className="text-xs text-gray-500 shrink-0">
              {share.view_count} {share.view_count === 1 ? 'view' : 'views'}
            </span>
          </button>
        );
      })}
    </div>
  )}
</div>
```

---

### 3. Memo Tab Updates

**File**: `apps/web/src/components/Dashboard/RightPreviewPane.tsx`

In the **Memo tab section** (around line 400-450), update:

```typescript
// BEFORE:
<p className="text-sm text-gray-600 mb-4">
  Investor-ready deal summaries with financials and key metrics
</p>
<button onClick={handleShareWithContacts}>Share with Contacts</button>

// AFTER:
{/* Remove subtitle paragraph */}
<div className="flex justify-end gap-2 mb-4">
  <button
    onClick={() => setShowCreateLiveShareModal(true)}
    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
  >
    Create Live Share
  </button>
  <button
    onClick={handleGenerateMemo}
    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
  >
    Generate Memo
  </button>
</div>
```

**Add state**:
```typescript
const [showCreateLiveShareModal, setShowCreateLiveShareModal] = useState(false);
```

**Remove**:
```typescript
import { ShareWithContactsModal } from '../Contacts/ShareWithContactsModal';
```

---

### 4. Create Live Share Modal

**File**: `apps/web/src/components/LiveShare/CreateLiveShareModal.tsx` (new file)

```tsx
import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createLiveShare } from '../../services/liveShareApi';
import { toast } from 'react-hot-toast';

interface CreateLiveShareModalProps {
  dealId: string;
  dealName: string;
  isOpen: boolean;
  onClose: () => void;
}

export function CreateLiveShareModal({
  dealId,
  dealName,
  isOpen,
  onClose,
}: CreateLiveShareModalProps) {
  const [expiresInDays, setExpiresInDays] = useState<7 | 30>(7);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: () => createLiveShare(dealId, expiresInDays),
    onSuccess: (data) => {
      setShareUrl(data.share_url);
      queryClient.invalidateQueries({ queryKey: ['liveShares'] });
      toast.success('Live share created!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Failed to create share');
    },
  });

  const handleCopyLink = () => {
    if (shareUrl) {
      navigator.clipboard.writeText(shareUrl);
      toast.success('Link copied to clipboard!');
    }
  };

  const handleClose = () => {
    setShareUrl(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-xl font-semibold mb-2">
          Share {dealName}
        </h2>
        
        {!shareUrl ? (
          <>
            <p className="text-sm text-gray-600 mb-4">
              Create a shareable link that anyone can view. You can track views
              and manage investor interest from your dashboard.
            </p>

            <div className="space-y-3 mb-6">
              <p className="text-sm font-medium text-gray-700">Link expires in:</p>
              
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="expiration"
                  checked={expiresInDays === 7}
                  onChange={() => setExpiresInDays(7)}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm text-gray-700">1 week</span>
              </label>

              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="expiration"
                  checked={expiresInDays === 30}
                  onChange={() => setExpiresInDays(30)}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm text-gray-700">1 month</span>
              </label>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={handleClose}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md"
              >
                Cancel
              </button>
              <button
                onClick={() => createMutation.mutate()}
                disabled={createMutation.isPending}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {createMutation.isPending ? 'Creating...' : 'Create Link'}
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-gray-600 mb-4">
              Your shareable link is ready!
            </p>

            <div className="bg-gray-50 p-3 rounded-md mb-4 break-all text-sm">
              {shareUrl}
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={handleClose}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md"
              >
                Close
              </button>
              <button
                onClick={handleCopyLink}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Copy Link
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
```

---

### 5. Interest Tracker Component

**File**: `apps/web/src/components/LiveShare/InterestTracker.tsx` (new file)

```tsx
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  getLiveShare,
  getInvestorInterest,
  type LiveShare,
  type InvestorInterest,
} from '../../services/liveShareApi';
import { AddInterestModal } from './AddInterestModal';
import { formatDistanceToNow } from 'date-fns';
import { toast } from 'react-hot-toast';

interface InterestTrackerProps {
  shareId: string;
}

export function InterestTracker({ shareId }: InterestTrackerProps) {
  const [showAddModal, setShowAddModal] = useState(false);

  const { data: share } = useQuery<LiveShare>({
    queryKey: ['liveShare', shareId],
    queryFn: () => getLiveShare(shareId),
  });

  const { data: interests = [] } = useQuery<InvestorInterest[]>({
    queryKey: ['interests', shareId],
    queryFn: () => getInvestorInterest(shareId),
    refetchInterval: 10000,
  });

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
    return <div className="p-6">Loading...</div>;
  }

  const expiresIn = formatDistanceToNow(new Date(share.expires_at), {
    addSuffix: true,
  });

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 p-6">
        <h2 className="text-xl font-semibold mb-1">Live Share Analytics</h2>
        <p className="text-sm text-gray-600">
          Track views and investor interest
        </p>
      </div>

      {/* Stats Bar */}
      <div className="border-b border-gray-200 p-6 bg-gray-50">
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <p className="text-xs text-gray-500 mb-1">Views</p>
            <p className="text-2xl font-semibold">{share.view_count}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Expires</p>
            <p className="text-sm font-medium">{expiresIn}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Status</p>
            <span
              className={`inline-flex px-2 py-1 text-xs font-medium rounded ${
                share.is_expired
                  ? 'bg-red-100 text-red-800'
                  : 'bg-green-100 text-green-800'
              }`}
            >
              {share.is_expired ? 'Expired' : 'Active'}
            </span>
          </div>
        </div>

        <button
          onClick={handleCopyLink}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
        >
          Copy Share Link
        </button>
      </div>

      {/* Investor Interest */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Investor Interest</h3>
          <button
            onClick={() => setShowAddModal(true)}
            className="px-3 py-1 bg-green-600 text-white rounded-md hover:bg-green-700 text-sm"
          >
            + Add
          </button>
        </div>

        {interests.length === 0 ? (
          <p className="text-sm text-gray-500">No interest entries yet</p>
        ) : (
          <div className="space-y-3">
            {interests.map((interest) => (
              <div
                key={interest.id}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex items-start justify-between mb-2">
                  <p className="font-medium">{interest.name}</p>
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${
                      interest.status === 'Interested'
                        ? 'bg-green-100 text-green-800'
                        : interest.status === 'Maybe'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {interest.status}
                  </span>
                </div>

                {interest.amount && (
                  <p className="text-lg font-semibold text-gray-900 mb-1">
                    ${interest.amount.toLocaleString()}
                  </p>
                )}

                {interest.notes && (
                  <p className="text-sm text-gray-600 mt-2">{interest.notes}</p>
                )}

                <p className="text-xs text-gray-400 mt-2">
                  {formatDistanceToNow(new Date(interest.created_at), {
                    addSuffix: true,
                  })}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Total */}
        {interests.length > 0 && totalAmount > 0 && (
          <div className="mt-6 pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">
                Total Interest
              </span>
              <span className="text-xl font-semibold text-gray-900">
                ${totalAmount.toLocaleString()}
              </span>
            </div>
          </div>
        )}
      </div>

      <AddInterestModal
        shareId={shareId}
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
      />
    </div>
  );
}
```

---

### 6. Add Interest Modal

**File**: `apps/web/src/components/LiveShare/AddInterestModal.tsx` (new file)

```tsx
import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { addInvestorInterest } from '../../services/liveShareApi';
import { toast } from 'react-hot-toast';

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
      queryClient.invalidateQueries({ queryKey: ['interests', shareId] });
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

    if (status !== 'Passed' && !amount) {
      toast.error('Amount is required for Interested/Maybe status');
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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-xl font-semibold mb-4">Add Investor Interest</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Investor name"
              required
            />
          </div>

          {/* Status */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Status *
            </label>
            <div className="space-y-2">
              {(['Interested', 'Maybe', 'Passed'] as InterestStatus[]).map((s) => (
                <label key={s} className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="radio"
                    name="status"
                    checked={status === s}
                    onChange={() => setStatus(s)}
                    className="w-4 h-4 text-blue-600"
                  />
                  <span className="text-sm text-gray-700">{s}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Amount */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Amount {status !== 'Passed' && '*'}
            </label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              disabled={status === 'Passed'}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
              placeholder="Investment amount"
              min="0"
              step="1000"
            />
          </div>

          {/* Notes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Notes (optional)
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Additional notes"
              rows={3}
            />
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={addMutation.isPending}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {addMutation.isPending ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

---

### 7. Public Share Page

**File**: `apps/web/src/pages/Share/PublicSharePage.tsx` (new file)

```tsx
import { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getPublicShare, trackView } from '../../services/liveShareApi';

export function PublicSharePage() {
  const { shareId } = useParams<{ shareId: string }>();

  const { data, isLoading, error } = useQuery({
    queryKey: ['publicShare', shareId],
    queryFn: () => getPublicShare(shareId!),
    enabled: !!shareId,
    retry: false,
  });

  // Track view on mount
  useEffect(() => {
    if (shareId) {
      trackView(shareId, undefined, navigator.userAgent).catch(console.error);
    }
  }, [shareId]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (error) {
    const errorMessage =
      (error as any)?.response?.status === 410
        ? 'This share link has expired.'
        : 'Share link not found or invalid.';

    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            {errorMessage}
          </h1>
          <p className="text-gray-600 mb-6">
            Please contact the deal owner for a new link.
          </p>
          <Link
            to="/"
            className="inline-block px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Go to ReFlow
          </Link>
        </div>
      </div>
    );
  }

  const { share, deal } = data!;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-blue-600">ReFlow</h1>
          <div className="text-sm text-gray-500">
            {share.view_count} {share.view_count === 1 ? 'view' : 'views'}
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Deal Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            {deal.property_name || 'Deal Details'}
          </h2>
          <p className="text-gray-600">{deal.address || ''}</p>
        </div>

        {/* Quick Stats */}
        {deal.purchase_price && (
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Purchase Price</p>
                <p className="text-xl font-semibold">
                  ${deal.purchase_price.toLocaleString()}
                </p>
              </div>
              {deal.noi && (
                <div>
                  <p className="text-sm text-gray-500">NOI</p>
                  <p className="text-xl font-semibold">
                    ${deal.noi.toLocaleString()}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Deal Summary */}
        {deal.summary && (
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Summary</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{deal.summary}</p>
          </div>
        )}

        {/* Documents */}
        {deal.documents && deal.documents.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Documents</h3>
            <div className="space-y-2">
              {deal.documents.map((doc: any) => (
                <div
                  key={doc.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded"
                >
                  <span className="text-sm text-gray-700">{doc.file_name}</span>
                  <span className="text-xs text-gray-500">
                    {doc.page_count || 0} pages
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Fixed Footer CTA */}
      <footer className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Want to create your own deal packages?
          </p>
          <Link
            to="/signup"
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium"
          >
            Sign Up Free
          </Link>
        </div>
      </footer>
    </div>
  );
}
```

---

### 8. Routing Integration

**File**: `apps/web/src/main.tsx`

Add public route (outside AuthGuard):

```typescript
// BEFORE:
const router = createBrowserRouter([
  {
    path: "/",
    element: <AuthGuard><DashboardThreePane /></AuthGuard>,
  },
  // ... other routes
]);

// AFTER:
import { PublicSharePage } from './pages/Share/PublicSharePage';

const router = createBrowserRouter([
  {
    path: "/share/:shareId",
    element: <PublicSharePage />, // No AuthGuard
  },
  {
    path: "/",
    element: <AuthGuard><DashboardThreePane /></AuthGuard>,
  },
  // ... other routes
]);
```

---

### 9. Right Preview Pane Integration

**File**: `apps/web/src/components/Dashboard/RightPreviewPane.tsx`

Update props and conditional rendering:

```typescript
// Add props:
interface RightPreviewPaneProps {
  // ... existing props
  selectedLiveShareId?: string | null;
}

// In component body:
export function RightPreviewPane({
  deal,
  onSelectTab,
  selectedTab,
  selectedLiveShareId, // NEW
  // ... other props
}: RightPreviewPaneProps) {
  
  // Early return if live share is selected
  if (selectedLiveShareId) {
    return <InterestTracker shareId={selectedLiveShareId} />;
  }
  
  // Rest of existing component...
}
```

**Add import**:
```typescript
import { InterestTracker } from '../LiveShare/InterestTracker';
```

---

### 10. Dashboard State Management

**File**: `apps/web/src/pages/Dashboard/DashboardThreePane.tsx`

Update state management:

```typescript
// Add state:
const [selectedLiveShareId, setSelectedLiveShareId] = useState<string | null>(null);

// Add handler:
const handleSelectLiveShare = (shareId: string) => {
  setSelectedLiveShareId(shareId);
  setSelectedDealId(null); // Clear deal selection
};

// When deal is selected, clear live share:
const handleSelectDeal = (dealId: string) => {
  setSelectedDealId(dealId);
  setSelectedLiveShareId(null); // Clear live share selection
};

// Pass to components:
<LeftNavPane
  // ... existing props
  onSelectLiveShare={handleSelectLiveShare}
  selectedLiveShareId={selectedLiveShareId}
/>

<RightPreviewPane
  // ... existing props
  selectedLiveShareId={selectedLiveShareId}
/>
```

---

### 11. Cleanup

**Remove**:
- `apps/web/src/components/Contacts/ShareWithContactsModal.tsx`
- Contact-related imports from `RightPreviewPane.tsx`
- Contact-related imports from `LeftNavPane.tsx`

**Search and remove**:
```bash
# Find all usages of ShareWithContactsModal
grep -r "ShareWithContactsModal" apps/web/src/

# Find all contact-related imports
grep -r "from.*ContactsApi" apps/web/src/
```

---

## Testing Checklist

### Manual Testing

1. **Create Live Share**:
   - [ ] Click "Create Live Share" button
   - [ ] Select 1 week expiration
   - [ ] Verify link is created
   - [ ] Copy link and open in incognito tab
   - [ ] Verify public page loads

2. **Sidebar**:
   - [ ] Verify "MY CONTACTS" section is removed
   - [ ] Verify "LIVE LINKS" section appears
   - [ ] Verify live links show green dot for active
   - [ ] Verify view count displays correctly
   - [ ] Click live link → verify Interest Tracker appears

3. **Interest Tracker**:
   - [ ] Verify view count updates
   - [ ] Click "+ Add" button
   - [ ] Add interest with "Interested" status and amount
   - [ ] Verify interest appears in list
   - [ ] Verify total amount calculates correctly

4. **Public Share Page**:
   - [ ] Open share link in incognito
   - [ ] Verify view count increments
   - [ ] Verify deal content displays
   - [ ] Try expired link → verify error message
   - [ ] Try invalid link → verify 404 message

---

## Success Criteria

- ✅ "MY CONTACTS" section completely removed
- ✅ "LIVE LINKS" section displays active shares with view counts
- ✅ Create live share in 2 clicks
- ✅ Public page loads without authentication
- ✅ View tracking works automatically
- ✅ Interest tracker displays stats and allows manual entries
- ✅ Copy link functionality works
- ✅ Expired links show appropriate message
- ✅ UI matches existing design patterns (Tailwind classes)

## Edge Cases Handled

- Empty states for no live shares
- Empty states for no interests
- Expired link UI (grayed out, no green dot)
- Invalid/deleted share (404 page)
- Network errors (toast notifications)
- Concurrent view tracking (backend handles atomicity)
