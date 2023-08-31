"use client"
import { useState } from "react";

const FAQItem = ({ index, question, answer, openIndex, handleClick }) => (
  <li>
    <button onClick={() => handleClick(index)} className="relative flex gap-2 items-center w-full py-5 text-base font-semibold text-left border-t md:text-lg border-base-content/10" aria-expanded={openIndex === index}>
      <span className="flex-1 text-base-content ">{question}</span>
      <svg className="flex-shrink-0 w-4 h-4 ml-auto fill-current" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
        <rect y="7" width="16" height="2" rx="1" className="transform origin-center transition duration-200 ease-out false"></rect>
        <rect y="7" width="16" height="2" rx="1" className="transform origin-center rotate-90 transition duration-200 ease-out false"></rect>
      </svg>
    </button>
    <div className={`transition-all duration-300 ease-in-out ${openIndex === index ? 'opacity-100 max-h-full' : 'opacity-0 max-h-0'}`} style={{transition: 'max-height 0.3s ease-in-out, opacity 0.3s ease-in-out'}}>
      <div className="pb-5 leading-relaxed">
        <div className="space-y-2 leading-relaxed">
          {answer}
        </div>
      </div>
    </div>
  </li>
);

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState(null);

  const handleClick = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  const faqItems = [
    {
      question: "How secure is the connection to my Google Sheet?",
      answer: "Our integration prioritizes security. We use OAuth 2.0 and never store your data."
    },
    {
      question: "Can I customize the expense categories?",
      answer: "Yes, our tool is flexible, allowing you to add or modify categories as you see fit."
    },
    {
      question: "Is there any manual intervention required?",
      answer: "While our AI is highly accurate, we provide an interface for you to make manual adjustments if needed."
    }
  ];

  return (
    <div className="py-24 px-8 max-w-5xl mx-auto flex flex-col md:flex-row gap-12">
      <div className="flex flex-col text-left basis-1/2">
        <p className="inline-block font-semibold text-primary mb-4">FAQ</p>
        <p className="sm:text-4xl text-3xl font-extrabold text-base-content">
          Frequently Asked Questions
        </p>
      </div>
      <ul className="basis-1/2">
        {faqItems.map((item, index) => (
          <FAQItem
            key={index}
            index={index}
            question={item.question}
            answer={item.answer}
            openIndex={openIndex}
            handleClick={handleClick}
          />
        ))}
      </ul>
    </div>
  );
}