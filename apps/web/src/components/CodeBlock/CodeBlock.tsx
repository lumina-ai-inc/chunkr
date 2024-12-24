// src/components/CodeBlock/CodeBlock.tsx
import Prism from "prismjs";
import { useEffect } from "react";
import "prismjs/themes/prism-tomorrow.css"; // or another theme
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-python";
import "prismjs/components/prism-bash";
import "prismjs/components/prism-rust";
import "./CodeBlock.css";

interface CodeBlockProps {
  code: string;
  language?: string;
  showLineNumbers?: boolean;
}

const CodeBlock = ({
  code,
  language = "typescript",
  showLineNumbers = true,
}: CodeBlockProps) => {
  useEffect(() => {
    Prism.highlightAll();
  }, [code, language]);

  return (
    <pre
      className={`code-content ${showLineNumbers ? "line-numbers" : ""}`}
      style={{
        background: "#ffffff00",
        borderRadius: "4px",
        padding: "0px",
        paddingTop: "16px",
      }}
    >
      <code className={`language-${language}`}>{code}</code>
    </pre>
  );
};

export default CodeBlock;
