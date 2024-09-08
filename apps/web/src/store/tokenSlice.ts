import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface TokenState {
  accessToken: string | null;
}

const initialState: TokenState = {
  accessToken: null,
};

const tokenSlice = createSlice({
  name: "token",
  initialState,
  reducers: {
    setAccessToken: (state, action: PayloadAction<string | null>) => {
      state.accessToken = action.payload;
    },
  },
});

export const { setAccessToken } = tokenSlice.actions;
export default tokenSlice.reducer;
