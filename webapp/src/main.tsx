import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import { ChatProvider } from "./hooks/useChat.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <ChatProvider>
    <App />
  </ChatProvider>
);