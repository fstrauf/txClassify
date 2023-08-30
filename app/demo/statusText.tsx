import React from "react";

interface StatusTextProps {
  text: string;
}

export default function StatusText({ text }: StatusTextProps) {
  return <p className="text-sm">{text}</p>;
}
