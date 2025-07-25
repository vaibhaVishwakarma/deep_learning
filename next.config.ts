import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';
const nextConfig: NextConfig = {
  output: 'export', // Required for `next export`
  basePath: isProd ? '/deep_learning' : '',
  assetPrefix: isProd ? '/deep_learning/' : '',
};

export default nextConfig;
