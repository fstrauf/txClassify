export default function Testimonials() {
  return (
    <div>
      <h2 className="text-2xl text-first mt-8">
        What Users Are Saying
      </h2>
      <div className="mt-4 space-y-6">
        <blockquote className="p-4 italic border-l-4 border-first">
          <p>
            "This tool transformed my monthly budgeting routine. It's efficient
            and incredibly accurate."
          </p>
          <cite className="mt-2 block text-right">- Jason N.</cite>
        </blockquote>
        <blockquote className="p-4 italic border-l-4 border-first">
          <p>
            "As a small business owner, the AI-driven classification saves me
            hours each month!"
          </p>
          <cite className="mt-2 block text-right">- Vicky S.</cite>
        </blockquote>
      </div>
    </div>
  );
}
