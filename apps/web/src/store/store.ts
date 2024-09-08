import { configureStore } from "@reduxjs/toolkit";
import pdfContentReducer from "./pdfContentSlice";
import userReducer from "./userSlice";
import tokenReducer from "./tokenSlice";

const store = configureStore({
  reducer: {
    pdfContent: pdfContentReducer,
    user: userReducer,
    token: tokenReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export default store;
