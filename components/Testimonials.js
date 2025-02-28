export default function Testimonials() {
  const testimonials = [
    {
      quote: "This tool transformed my monthly budgeting routine. It's efficient and incredibly accurate.",
      author: "Jason N.",
      role: "Personal Finance Enthusiast"
    },
    {
      quote: "As a small business owner, the AI-driven classification saves me hours each month!",
      author: "Vicky S.",
      role: "Business Owner"
    },
    {
      quote: "The Google Sheets™ integration is seamless. It just works with my existing setup.",
      author: "Michael R.",
      role: "Freelancer"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      {testimonials.map((testimonial, index) => (
        <div
          key={index}
          className="relative bg-white p-6 rounded-xl shadow-soft"
        >
          <div className="absolute -top-4 left-6 text-primary-light text-5xl">"</div>
          <div className="pt-4">
            <p className="text-gray-700 mb-4">
              {testimonial.quote}
            </p>
            <div className="border-t border-gray-100 pt-4">
              <p className="font-semibold text-gray-900">{testimonial.author}</p>
              <p className="text-sm text-gray-500">{testimonial.role}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
