"use client";

import dynamic from "next/dynamic";

const BetaAccessBanner = dynamic(() => import("./BetaAccessBanner"), {
  ssr: false,
});

export default function BannerWrapper() {
  return <BetaAccessBanner />;
}
