import { NextRequest, NextResponse } from "next/server";

export async function GET(_req: NextRequest) {
  return NextResponse.json(
    {
      error: "not_implemented",
      message: "Endpoint not implemented yet.",
    },
    { status: 501 }
  );
}
