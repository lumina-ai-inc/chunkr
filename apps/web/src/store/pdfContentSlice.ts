import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { BoundingBoxes } from "../models/chunk.model";

interface PdfContentState {
  content: BoundingBoxes;
  isLoading: boolean;
  error: string | null;
}

const initialState: PdfContentState = {
  content: [],
  isLoading: false,
  error: null,
};

const pdfContentSlice = createSlice({
  name: "pdfContent",
  initialState,
  reducers: {
    setPdfContent: (state, action: PayloadAction<BoundingBoxes>) => {
      state.content = action.payload;
      state.isLoading = false;
      state.error = null;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
      state.isLoading = false;
    },
  },
});

export const { setPdfContent, setLoading, setError } = pdfContentSlice.actions;
export default pdfContentSlice.reducer;
