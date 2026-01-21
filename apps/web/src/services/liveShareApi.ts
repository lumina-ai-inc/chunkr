import axiosInstance from "./axios.config";
import { isMockDeal } from "./mockDealData";
import {
  MOCK_LIVE_SHARES,
  MOCK_SHARE_VIEWS,
  MOCK_INVESTOR_INTEREST,
  getMockDealForShare,
  saveMockLiveShareData,
} from "./mockLiveShareData";

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
  // Use mock data if deal is a mock deal, otherwise use real backend
  if (isMockDeal(dealId)) {
    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 500));
    
    const shortId = `share${Math.random().toString(36).substring(2, 10)}`;
    const origin = typeof window !== 'undefined' ? window.location.origin : '';
    const newShare: LiveShare = {
      id: `live-share-${Date.now()}`,
      deal_id: dealId,
      short_id: shortId,
      share_url: `${origin}/share/${shortId}`,
      expires_at: new Date(Date.now() + expiresInDays * 24 * 60 * 60 * 1000).toISOString(),
      view_count: 0,
      created_at: new Date().toISOString(),
      is_expired: false,
    };
    
    // Add to mock data and persist
    MOCK_LIVE_SHARES.push(newShare);
    saveMockLiveShareData();
    return newShare;
  }
  
  const response = await axiosInstance.post('/api/v1/live-shares', {
    deal_id: dealId,
    expires_in_days: expiresInDays,
  });
  return response.data;
};

export const getLiveShares = async (): Promise<LiveShare[]> => {
  // Always try real backend first, fall back to mock data if needed
  try {
    const response = await axiosInstance.get('/api/v1/live-shares');
    return response.data;
  } catch (error) {
    // If backend fails, use mock data (for mock deals or when backend is unavailable)
    await new Promise((resolve) => setTimeout(resolve, 300));
    // Update is_expired for each share before returning
    const now = new Date();
    return MOCK_LIVE_SHARES.filter((share) => isMockDeal(share.deal_id)).map((share) => ({
      ...share,
      is_expired: new Date(share.expires_at) < now,
    }));
  }
};

export const getLiveShare = async (shareId: string): Promise<LiveShare> => {
  // Check if share exists in mock data first
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    // Update is_expired before returning
    const now = new Date();
    return {
      ...mockShare,
      is_expired: new Date(mockShare.expires_at) < now,
    };
  }
  
  // Try real backend
  try {
    const response = await axiosInstance.get(`/api/v1/live-shares/${shareId}`);
    return response.data;
  } catch (error) {
    // If not found in backend and not in mock, throw error
    if (mockShare) {
      const now = new Date();
      return {
        ...mockShare,
        is_expired: new Date(mockShare.expires_at) < now,
      };
    }
    throw error;
  }
};

export const getShareViews = async (shareId: string): Promise<ShareView[]> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    return MOCK_SHARE_VIEWS[shareId] || [];
  }
  
  const response = await axiosInstance.get(`/api/v1/live-shares/${shareId}/views`);
  return response.data;
};

export const addInvestorInterest = async (
  shareId: string,
  interest: CreateInterestRequest
): Promise<InvestorInterest> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    
    const newInterest: InvestorInterest = {
      id: `interest-${Date.now()}`,
      live_share_id: shareId,
      name: interest.name,
      amount: interest.amount,
      status: interest.status,
      notes: interest.notes,
      created_at: new Date().toISOString(),
    };
    
    // Add to mock data and persist
    if (!MOCK_INVESTOR_INTEREST[shareId]) {
      MOCK_INVESTOR_INTEREST[shareId] = [];
    }
    MOCK_INVESTOR_INTEREST[shareId].push(newInterest);
    saveMockLiveShareData();
    return newInterest;
  }
  
  const response = await axiosInstance.post(
    `/api/v1/live-shares/${shareId}/interest`,
    interest
  );
  return response.data;
};

export const getInvestorInterest = async (
  shareId: string
): Promise<InvestorInterest[]> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    return MOCK_INVESTOR_INTEREST[shareId] || [];
  }
  
  const response = await axiosInstance.get(`/api/v1/live-shares/${shareId}/interest`);
  return response.data;
};

export const deleteInvestorInterest = async (
  shareId: string,
  interestId: string
): Promise<void> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    if (MOCK_INVESTOR_INTEREST[shareId]) {
      MOCK_INVESTOR_INTEREST[shareId] = MOCK_INVESTOR_INTEREST[shareId].filter(
        (i) => i.id !== interestId
      );
      saveMockLiveShareData();
    }
    return;
  }
  
  await axiosInstance.delete(`/api/v1/live-shares/${shareId}/interest/${interestId}`);
};

// Delete a live share (for cleanup)
export const deleteLiveShare = async (shareId: string): Promise<void> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    
    // Remove from mock data
    const index = MOCK_LIVE_SHARES.findIndex((s) => s.id === shareId);
    if (index !== -1) {
      MOCK_LIVE_SHARES.splice(index, 1);
    }
    
    // Clean up related data
    delete MOCK_SHARE_VIEWS[shareId];
    delete MOCK_INVESTOR_INTEREST[shareId];
    
    saveMockLiveShareData();
    return;
  }
  
  await axiosInstance.delete(`/api/v1/live-shares/${shareId}`);
};

// Notify watchers with a message
export const notifyWatchers = async (
  shareId: string,
  message: string
): Promise<void> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.id === shareId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    // Mock implementation - just log the notification
    console.log(`[Mock] Notifying watchers for share ${shareId}: ${message}`);
    return;
  }
  
  await axiosInstance.post(`/api/v1/live-shares/${shareId}/notify`, {
    message,
  });
};

// ============================================================================
// PUBLIC ENDPOINTS (no auth required)
// ============================================================================

export const getPublicShare = async (
  shortId: string
): Promise<{ share: LiveShare; deal: any }> => {
  // Check if share exists in mock data
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.short_id === shortId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    
    // Check if expired
    const now = new Date();
    const isExpired = new Date(mockShare.expires_at) < now;
    if (isExpired) {
      const error: any = new Error('Share has expired');
      error.response = { status: 410 };
      throw error;
    }
    
    const deal = getMockDealForShare(mockShare.deal_id);
    if (!deal) {
      throw new Error('Deal not found');
    }
    
    return {
      share: {
        ...mockShare,
        is_expired: false,
      },
      deal,
    };
  }
  
  // Try real backend
  try {
    const response = await axiosInstance.get(`/api/v1/public/share/${shortId}`);
    return response.data;
  } catch (error) {
    // If not found in backend and not in mock, throw error
    if (mockShare) {
      const deal = getMockDealForShare(mockShare.deal_id);
      if (deal) {
        const now = new Date();
        const isExpired = new Date(mockShare.expires_at) < now;
        if (isExpired) {
          const expiredError: any = new Error('Share has expired');
          expiredError.response = { status: 410 };
          throw expiredError;
        }
        return {
          share: {
            ...mockShare,
            is_expired: false,
          },
          deal,
        };
      }
    }
    throw error;
  }
};

export const trackView = async (
  shortId: string,
  ipAddress?: string,
  userAgent?: string
): Promise<void> => {
  // Check if share is a mock share
  const mockShare = MOCK_LIVE_SHARES.find((s) => s.short_id === shortId);
  if (mockShare && isMockDeal(mockShare.deal_id)) {
    // Simulate view tracking - increment count and add view record
    mockShare.view_count += 1;
    
    // Add view record
    if (!MOCK_SHARE_VIEWS[mockShare.id]) {
      MOCK_SHARE_VIEWS[mockShare.id] = [];
    }
    MOCK_SHARE_VIEWS[mockShare.id].unshift({
      id: `view-${Date.now()}`,
      live_share_id: mockShare.id,
      viewed_at: new Date().toISOString(),
      ip_address: ipAddress,
      user_agent: userAgent,
    });
    
    saveMockLiveShareData();
    return;
  }
  
  // Try real backend
  try {
    await axiosInstance.post(`/api/v1/public/share/${shortId}/view`, {
      ip_address: ipAddress,
      user_agent: userAgent,
    });
  } catch (error) {
    // If backend fails but it's a mock share, we already handled it above
    if (!mockShare) {
      throw error;
    }
  }
};
