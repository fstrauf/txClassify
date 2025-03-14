import React from "react";

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  className?: string;
}

export default function PageHeader({ title, subtitle, className = "" }: PageHeaderProps) {
  return (
    <div className={`text-center ${className}`}>
      <h1 className="text-4xl md:text-6xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-primary-dark via-primary to-secondary animate-gradient pb-2 leading-normal">
        {title}
      </h1>
      {subtitle && <p className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto mb-12">{subtitle}</p>}
    </div>
  );
}
