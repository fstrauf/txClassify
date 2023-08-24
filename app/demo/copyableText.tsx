import React, { useState } from "react";


export default function CopyableText(props: {textToCopy: string}) {
//   const textToCopy = "expense-sorted@txclassify.iam.gserviceaccount.com";
  const [isCopied, setIsCopied] = useState(false);

  const handleCopyClick = () => {
    navigator.clipboard.writeText(props.textToCopy);
    setIsCopied(true);

    // Reset the "Copied" state after a brief delay
    setTimeout(() => {
      setIsCopied(false);
    }, 1500);
  };

  return (
    <span className="bg-first text-white cursor-pointer mr-1" onClick={handleCopyClick}>
      {isCopied ? "Copied!" : props.textToCopy}
    </span>
  );
}
