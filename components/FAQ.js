"use client"
import { useState } from "react";

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState(null);

  const faqs = [
    {
      question: "How does the AI categorization work?",
      answer: "Our AI model learns from your existing categorized expenses. When you submit new transactions, it analyzes patterns in descriptions and amounts to automatically assign the most appropriate category."
    },
    {
      question: "Is my data secure?",
      answer: "Yes! Your data stays in your Google Sheets™. We only process the specific transactions you send for categorization, and we don't store any of your financial data permanently."
    },
    {
      question: "How accurate is the categorization?",
      answer: "The accuracy depends on your training data. With a good set of pre-categorized transactions, our AI typically achieves 90%+ accuracy. The more you use it, the better it gets."
    },
    {
      question: "Can I customize the categories?",
      answer: "Absolutely! You can train the model with your own custom categories. The AI will learn your specific categorization preferences and apply them to new transactions."
    }
  ];

  return (
    <div className="max-w-3xl mx-auto divide-y divide-gray-200">
      {faqs.map((faq, index) => (
        <div key={index} className="py-6">
          <button
            onClick={() => setOpenIndex(openIndex === index ? null : index)}
            className="flex justify-between items-center w-full text-left"
          >
            <h3 className="text-lg font-semibold text-gray-900">
              {faq.question}
            </h3>
            <span className="ml-6 flex-shrink-0">
              <svg
                className={`w-6 h-6 transform ${openIndex === index ? 'rotate-180' : ''} text-gray-400`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </span>
          </button>
          {openIndex === index && (
            <div className="mt-4">
              <p className="text-gray-600">
                {faq.answer}
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}