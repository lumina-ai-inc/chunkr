import { configureStore } from "@reduxjs/toolkit";
import pdfContentReducer from "./pdfContentSlice";

const store = configureStore({
  reducer: {
    pdfContent: pdfContentReducer,
    // Add other reducers here if you have any
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export default store;
