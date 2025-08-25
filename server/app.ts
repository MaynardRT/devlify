import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Types
interface ChatMessage {
  role: string;
  content: string;
  name?: string;
}

interface ChatRequest {
  messages: ChatMessage[];
  preferredApi?: "google" | "deepseek";
}

// Load environment variables
dotenv.config({
  path: [".env.local", ".env"],
  debug: process.env.DEBUG === "true",
});

// Validate environment variables
if (!process.env.GG_API_KEY && !process.env.DS_API_KEY) {
  throw new Error(
    "At least one API key (GG_API_KEY or DS_API_KEY) must be configured"
  );
}

// Initialize API clients
const deepseekClient = process.env.DS_API_KEY
  ? new OpenAI({
      baseURL: "https://api.deepseek.com/v1",
      apiKey: process.env.DS_API_KEY,
    })
  : null;

const googleAI = process.env.GG_API_KEY
  ? new GoogleGenerativeAI(process.env.GG_API_KEY)
  : null;
const googleModel = googleAI?.getGenerativeModel({ model: "gemini-2.0-flash" });

const app = express();
app.use(express.json());

// CORS configuration - fixed to use environment variable or default
const corsOptions = {
  origin:
    process.env.CLIENT_ORIGIN ||
    (process.env.NODE_ENV === "production"
      ? "your-production-domain"
      : "http://localhost:5173"),
  methods: ["POST"],
  allowedHeaders: ["Content-Type"],
};

app.use(cors(corsOptions));

// API Selection Logic - Fixed Google API formatting
async function tryAPI(
  apiName: "deepseek" | "google",
  messages: ChatMessage[]
): Promise<string | null> {
  try {
    if (apiName === "google" && googleModel) {
      // Format messages for Google AI - fixed to use proper chat format
      const formattedMessages = messages.map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
      }));

      // Start a chat session with history
      const chat = await googleModel.startChat({
        history: formattedMessages.slice(0, -1),
      });

      // Send the last message
      const result = await chat.sendMessage(
        formattedMessages[formattedMessages.length - 1].parts[0].text
      );
      const response = await result.response;
      return response.text();
    }

    if (apiName === "deepseek" && deepseekClient) {
      // Format messages for DeepSeek - removed redundant system message
      const completion = await deepseekClient.chat.completions.create({
        messages: messages.map((msg) => {
          // Only include 'name' if it's defined, and cast role to allowed types
          const { role, content, name } = msg;
          const allowedRoles = ["system", "user", "assistant"] as const;
          if (!allowedRoles.includes(role as any)) {
            throw new Error(`Invalid role: ${role}`);
          }
          const base = {
            role: role as "system" | "user" | "assistant",
            content,
          };
          return name ? { ...base, name } : base;
        }),
        model: "deepseek-chat",
      });
      return completion.choices[0]?.message?.content ?? null;
    }

    return null;
  } catch (error) {
    console.error(
      `${apiName} API Error:`,
      error instanceof Error ? error.message : String(error)
    );
    return null;
  }
}

app.post("/api/chat", async (req, res) => {
  try {
    const { messages, preferredApi = "google" } = req.body as ChatRequest;

    if (!messages?.length) {
      return res.status(400).json({
        error: "Invalid or missing 'messages' field.",
      });
    }

    // Validate message format
    const isValidMessages = messages.every(
      (msg) => typeof msg.role === "string" && typeof msg.content === "string"
    );

    if (!isValidMessages) {
      return res.status(400).json({
        error:
          "Invalid message format. Each message must have 'role' and 'content' as strings.",
      });
    }

    let reply: string | null = null;
    let usedApi = "";

    // Try preferred API first if available
    if (preferredApi === "google" && process.env.GG_API_KEY) {
      reply = await tryAPI("google", messages);
      if (reply) usedApi = "google";
    } else if (preferredApi === "deepseek" && process.env.DS_API_KEY) {
      reply = await tryAPI("deepseek", messages);
      if (reply) usedApi = "deepseek";
    }

    // Try fallback if preferred API failed or isn't available
    if (!reply) {
      if (preferredApi !== "google" && process.env.GG_API_KEY) {
        reply = await tryAPI("google", messages);
        if (reply) usedApi = "google";
      } else if (preferredApi !== "deepseek" && process.env.DS_API_KEY) {
        reply = await tryAPI("deepseek", messages);
        if (reply) usedApi = "deepseek";
      }
    }

    if (!reply) {
      return res.status(503).json({
        error: "All available API attempts failed",
        message: "Service temporarily unavailable",
      });
    }

    res.status(200).json({
      success: true,
      message: "Response generated successfully.",
      reply,
      usedApi,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error("API Error:", {
      message: error.message,
      stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
    });

    res.status(error.status || 500).json({
      error: "API request failed",
      message: error.message || "Unknown error occurred",
    });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log("Environment:", process.env.NODE_ENV || "development");
  console.log(
    "Available APIs:",
    [process.env.GG_API_KEY && "Google", process.env.DS_API_KEY && "DeepSeek"]
      .filter(Boolean)
      .join(", ")
  );
});
