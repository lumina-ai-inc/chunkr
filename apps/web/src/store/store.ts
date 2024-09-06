import { configureStore } from "@reduxjs/toolkit";
import pdfContentReducer from "./pdfContentSlice";
import userReducer from "./userSlice";

const store = configureStore({
  reducer: {
    pdfContent: pdfContentReducer,
    user: userReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export default store;
