export default function FAQ() {
  return (
    <div>
      <h2 className="text-2xl text-first mt-8">
        Frequently Asked Questions
      </h2>
      <div className="flex flex-col gap-3 mt-4 space-y-4 text-lg">
        <div>
          <h3 className="font-semibold">
            How secure is the connection to my Google Sheet?
          </h3>
          <p>
            Our integration prioritizes security. We use OAuth 2.0 and never
            store your data.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">
            Can I customize the expense categories?
          </h3>
          <p>
            Yes, our tool is flexible, allowing you to add or modify categories
            as you see fit.
          </p>
        </div>
        <div>
          <h3 className="font-semibold">
            Is there any manual intervention required?
          </h3>
          <p>
            While our AI is highly accurate, we provide an interface for you to
            make manual adjustments if needed.
          </p>
        </div>
      </div>
    </div>
  );
}
