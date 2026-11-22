import express from "express";
import dotenv from "dotenv";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from "openai";
import session from "express-session";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const dir = path.dirname(fileURLToPath(import.meta.url));

const baseUrl = () =>
  (process.env.APP_BASE_URL || `http://localhost:${port}`).replace(/\/$/, "");

app.set("trust proxy", 1);
app.use(express.json({ limit: "2mb" }));

app.use(session({
  secret: process.env.SESSION_SECRET || "replace-this-with-a-long-random-secret",
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    sameSite: "lax",
    secure: "auto",
    maxAge: 86400000
  }
}));

app.use(express.static(path.join(dir, "../public")));

/* ================= SYSTEM ================= */

const mainPrompt = `
You are NOVAAA AI, an intelligent creative assistant.

Your responsibilities:
- Story writing
- Horror and mystery writing
- Screenplays
- Character creation
- World building
- YouTube content
- Instagram content
- Video concepts
- Animation planning
- Creator tools
- Content production
- Creative brainstorming

Be creative, accurate and helpful.
Follow the user's requested language.
Never pretend to perform actions you cannot perform.
`;

const features = [
  "AI Chat",
  "Story Generator",
  "Script Writer",
  "Horror Story Builder",
  "Dialogue Generator",
  "Character Creator",
  "Character Backstory",
  "Plot Generator",
  "Plot Twist Generator",
  "Scene Generator",
  "Episode Planner",
  "Series Planner",
  "YouTube Ideas",
  "YouTube Titles",
  "YouTube Descriptions",
  "YouTube Tags",
  "YouTube Hooks",
  "Shorts Ideas",
  "Instagram Ideas",
  "Instagram Captions",
  "Creator Prompts",
  "Image Prompts",
  "Video Prompts",
  "Voiceover Scripts",
  "Thumbnail Ideas",
  "Content Planner",
  "Project Memory",
  "Project Notes",
  "Creative Brainstorm",
  "Rewrite Assistant",
  "Tone Changer",
  "Dialogue Enhancer",
  "Story Expander",
  "Story Shortener",
  "Story Continuation",
  "Ending Generator",
  "Opening Hook Generator",
  "Suspense Builder",
  "Mystery Builder",
  "Horror Builder",
  "Comedy Builder",
  "Action Builder",
  "Fantasy Builder",
  "Sci-Fi Builder",
  "Drama Builder",
  "Animation Planner",
  "Animatic Planner",
  "Shot List",
  "Scene Breakdown",
  "Camera Direction",
  "Lighting Direction",
  "Sound Design",
  "Music Direction",
  "Voice Direction",
  "Character Voice Guide",
  "Location Generator",
  "Prop Generator",
  "Costume Generator",
  "World Builder",
  "Lore Generator",
  "Episode Recap",
  "Episode Preview",
  "Trailer Script",
  "Teaser Script",
  "Cliffhanger Generator",
  "Comment Reply Ideas",
  "Community Post Ideas",
  "Poll Ideas",
  "Trend Brainstorm",
  "Content Calendar",
  "Upload Checklist",
  "Creator Checklist",
  "Production Checklist",
  "SEO Assistant",
  "Keyword Ideas",
  "Hashtag Ideas",
  "Brand Name Generator",
  "Channel Name Generator",
  "Series Name Generator",
  "Character Name Generator",
  "Video Concept Generator",
  "Short Film Generator",
  "Film Treatment",
  "Logline Generator",
  "Synopsis Generator",
  "Pitch Generator",
  "Scene Dialogue",
  "Monologue Generator",
  "Narration Generator",
  "First Person Rewrite",
  "Third Person Rewrite",
  "Hindi Script",
  "English Script",
  "Hinglish Script",
  "Translation Assistant",
  "Grammar Assistant",
  "Summarizer",
  "Idea Organizer",
  "Project Dashboard",
  "YouTube Connection",
  "Instagram Connection",
  "Social Status",
  "Provider Status",
  "Command Centre",
  "NOVAAA Assistant",
  "Creative Director",
  "Production Assistant",
  "Content Assistant",
  "Project Assistant",
  "Feature Router",
  "Workspace Manager",
  "Memory Manager",
  "Safety Filter",
  "System Health"
];

/* ================= PROVIDERS ================= */

const providerReady = () => ({
  gemini: !!process.env.GEMINI_API_KEY,
  claude: !!process.env.ANTHROPIC_API_KEY,
  openai: !!process.env.OPENAI_API_KEY
});

/* GEMINI */

async function askGemini(message, systemPrompt) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("Gemini API key is missing.");
  }

  const client = new GoogleGenerativeAI(
    process.env.GEMINI_API_KEY
  );

  const model = client.getGenerativeModel({
    model: process.env.GEMINI_MODEL || "gemini-3.8-flash",
    systemInstruction: systemPrompt
  });

  let lastError;

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const result = await model.generateContent(message);
      return result.response.text();
    } catch (error) {
      lastError = error;

      const temporary =
        /503|429|500|502|504|overload|high demand/i.test(
          String(error.message)
        );

      if (!temporary || attempt === 2) break;

      await new Promise(resolve =>
        setTimeout(resolve, 1000 * (2 ** attempt))
      );
    }
  }

  throw lastError;
}

/* CLAUDE */

async function askClaude(message, systemPrompt) {
  const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY
  });

  const response = await client.messages.create({
    model: process.env.CLAUDE_MODEL ||
      "claude-3-5-sonnet-20241022",
    max_tokens: 2000,
    system: systemPrompt,
    messages: [
      { role: "user", content: message }
    ]
  });

  return response.content
    .filter(item => item.type === "text")
    .map(item => item.text)
    .join("\n");
}

/* OPENAI */

async function askOpenAI(message, systemPrompt) {
  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
  });

  const response = await client.chat.completions.create({
    model: process.env.OPENAI_MODEL || "gpt-4o-mini",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: message }
    ]
  });

  return response.choices[0]?.message?.content || "";
}

/* PROVIDER ROUTER */

async function askProvider(provider, message, systemPrompt) {
  if (provider === "gemini") {
    return askGemini(message, systemPrompt);
  }

  if (provider === "claude") {
    return askClaude(message, systemPrompt);
  }

  if (provider === "openai") {
    return askOpenAI(message, systemPrompt);
  }

  throw new Error("Unknown AI provider.");
}

/* ================= RESPONSE COMPATIBILITY ================= */

function sendAIResult(res, provider, answer, extra = {}) {
  return res.json({
    ok: true,
    provider,
    answer,
    reply: answer,
    result: answer,
    response: answer,
    message: answer,
    ...extra
  });
}

/* ================= STATUS ================= */

app.get("/api/health", (_, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    server: "online",
    version: "3.1.0",
    timestamp: new Date().toISOString()
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    server: "online",
    version: "3.1.0",
    providers: providerReady(),
    connections: {
      youtube: !!req.session.youtubeConnected,
      instagram: false
    },
    featureCount: features.length
  });
});

app.get("/api/providers", (_, res) => {
  res.json({
    ok: true,
    providers: providerReady()
  });
});

app.get("/api/features", (_, res) => {
  res.json({
    ok: true,
    count: features.length,
    features
  });
});

/* ================= AI CHAT ================= */

app.post("/api/chat", async (req, res) => {
  try {
    const {
      message,
      provider = "gemini",
      systemPrompt = "",
      feature = "",
      history = []
    } = req.body || {};

    if (!message || typeof message !== "string") {
      return res.status(400).json({
        ok: false,
        error: "Please enter a message."
      });
    }

    if (message.length > 15000) {
      return res.status(400).json({
        ok: false,
        error: "Message is too long."
      });
    }

    const instruction = [
      mainPrompt,
      feature ? `Current tool: ${feature}` : "",
      systemPrompt
    ].filter(Boolean).join("\n\n");

    const recent = Array.isArray(history)
      ? history.slice(-8).map(item => {
          const role = item?.role === "assistant"
            ? "Assistant"
            : "User";

          return `${role}: ${String(item?.content || "").slice(0, 3000)}`;
        }).join("\n")
      : "";

    const prompt = recent
      ? `Recent conversation:\n${recent}\n\nUser:\n${message}`
      : message;

    const answer = await askProvider(
      provider,
      prompt,
      instruction
    );

    sendAIResult(res, provider, answer);

  } catch (error) {
    console.error("AI CHAT ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "AI request failed."
    });
  }
});

/* ================= COMMAND CENTRE ================= */

app.post("/api/command", async (req, res) => {
  try {
    const {
      command,
      input = "",
      prompt = "",
      provider = "gemini",
      feature = ""
    } = req.body || {};

    const task = String(command || prompt || "").trim();
    const userInput = String(input || "").trim();

    if (!task && !userInput) {
      return res.status(400).json({
        ok: false,
        error: "Enter a command or prompt."
      });
    }

    const fullPrompt = `
NOVAAA CREATIVE TASK: ${feature || task}

Task:
${task}

User input:
${userInput}

Complete the task directly.
Use headings where useful.
Follow the user's requested language.
`;

    const answer = await askProvider(
      provider,
      fullPrompt,
      mainPrompt
    );

    sendAIResult(res, provider, answer, {
      command: task
    });

  } catch (error) {
    console.error("COMMAND ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "Command failed."
    });
  }
});

/* ================= PROJECT MEMORY ================= */

const memoryBySession = new Map();

function getMemories(req) {
  if (!memoryBySession.has(req.sessionID)) {
    memoryBySession.set(req.sessionID, []);
  }

  return memoryBySession.get(req.sessionID);
}

app.get("/api/memory", (req, res) => {
  const memories = getMemories(req);

  res.json({
    ok: true,
    memories,
    projects: memories
  });
});

app.post("/api/memory", (req, res) => {
  const { title, content } = req.body || {};

  if (!title || !content) {
    return res.status(400).json({
      ok: false,
      error: "Title and content are required."
    });
  }

  const memories = getMemories(req);

  const memory = {
    id: crypto.randomUUID(),
    title: String(title).slice(0, 160),
    content: String(content).slice(0, 10000),
    createdAt: new Date().toISOString()
  };

  memories.unshift(memory);

  res.json({
    ok: true,
    memory,
    memories
  });
});

app.delete("/api/memory/:id", (req, res) => {
  const memories = getMemories(req);

  const updated = memories.filter(
    item => item.id !== req.params.id
  );

  memoryBySession.set(req.sessionID, updated);

  res.json({
    ok: true,
    memories: updated
  });
});

/* PROJECT COMPATIBILITY */

app.get("/api/projects", (req, res) => {
  res.json({
    ok: true,
    projects: getMemories(req)
  });
});

app.post("/api/projects", (req, res) => {
  const { name, content = "" } = req.body || {};

  if (!name) {
    return res.status(400).json({
      ok: false,
      error: "Project name is required."
    });
  }

  const projects = getMemories(req);

  const project = {
    id: crypto.randomUUID(),
    name: String(name).slice(0, 160),
    title: String(name).slice(0, 160),
    content: String(content).slice(0, 10000),
    createdAt: new Date().toISOString()
  };

  projects.unshift(project);

  res.json({
    ok: true,
    project,
    projects
  });
});

app.delete("/api/projects/:id", (req, res) => {
  const updated = getMemories(req).filter(
    item => item.id !== req.params.id
  );

  memoryBySession.set(req.sessionID, updated);

  res.json({
    ok: true,
    projects: updated
  });
});

/* ================= YOUTUBE ================= */

const youtubeReady = () =>
  !!(
    process.env.GOOGLE_CLIENT_ID &&
    process.env.GOOGLE_CLIENT_SECRET
  );

const youtubeRedirectUri = () =>
  `${baseUrl()}/auth/youtube/callback`;

app.get("/auth/youtube", async (req, res) => {
  if (!youtubeReady()) {
    return res.status(400).send(
      "YouTube OAuth is not configured."
    );
  }

  const state = crypto.randomBytes(24).toString("hex");

  req.session.youtubeOAuthState = state;

  try {
    await new Promise((resolve, reject) => {
      req.session.save(error =>
        error ? reject(error) : resolve()
      );
    });

    const params = new URLSearchParams({
      client_id: process.env.GOOGLE_CLIENT_ID,
      redirect_uri: youtubeRedirectUri(),
      response_type: "code",
      scope:
        "https://www.googleapis.com/auth/youtube.readonly " +
        "https://www.googleapis.com/auth/youtube.upload",
      access_type: "offline",
      prompt: "consent",
      state
    });

    res.redirect(
      `https://accounts.google.com/o/oauth2/v2/auth?${params}`
    );

  } catch (error) {
    console.error(error);
    res.status(500).send("Could not start YouTube connection.");
  }
});

app.get("/auth/youtube/callback", async (req, res) => {
  try {
    const { code, state, error } = req.query;

    if (error) {
      return res.redirect("/?youtube=cancelled");
    }

    if (
      !state ||
      state !== req.session.youtubeOAuthState
    ) {
      return res.status(403).send(
        "Invalid OAuth state. Try connecting again."
      );
    }

    delete req.session.youtubeOAuthState;

    if (!code) {
      return res.status(400).send("Missing authorisation code.");
    }

    const response = await fetch(
      "https://oauth2.googleapis.com/token",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded"
        },
        body: new URLSearchParams({
          code,
          client_id: process.env.GOOGLE_CLIENT_ID,
          client_secret: process.env.GOOGLE_CLIENT_SECRET,
          redirect_uri: youtubeRedirectUri(),
          grant_type: "authorization_code"
        })
      }
    );

    const tokens = await response.json();

    if (!response.ok) {
      console.error("GOOGLE TOKEN ERROR:", tokens);

      return res.status(400).send(
        "Google token exchange failed."
      );
    }

    req.session.youtubeConnected = true;

    await new Promise((resolve, reject) => {
      req.session.save(error =>
        error ? reject(error) : resolve()
      );
    });

    res.redirect("/?youtube=connected");

  } catch (error) {
    console.error("YOUTUBE CALLBACK:", error);
    res.status(500).send("YouTube connection failed.");
  }
});

app.get("/api/youtube/status", (req, res) => {
  res.json({
    ok: true,
    connected: !!req.session.youtubeConnected,
    configured: youtubeReady()
  });
});

app.post("/api/youtube/disconnect", (req, res) => {
  delete req.session.youtubeConnected;

  req.session.save(error => {
    if (error) {
      return res.status(500).json({
        ok: false,
        error: "Could not disconnect YouTube."
      });
    }

    res.json({
      ok: true,
      connected: false
    });
  });
});

/* ================= INSTAGRAM ================= */

app.get("/api/instagram/status", (_, res) => {
  const configured = !!(
    process.env.INSTAGRAM_APP_ID &&
    process.env.INSTAGRAM_APP_SECRET
  );

  res.json({
    ok: true,
    configured,
    connected: false,
    message: configured
      ? "Credentials detected. OAuth still needs implementation."
      : "Instagram credentials are not configured."
  });
});

app.get("/auth/instagram", (_, res) => {
  res.status(501).send(
    "Instagram OAuth has not been implemented yet."
  );
});

/* ================= SOCIAL STATUS ================= */

app.get("/api/social/status", (req, res) => {
  res.json({
    ok: true,
    youtube: !!req.session.youtubeConnected,
    youtubeConnected: !!req.session.youtubeConnected,
    instagram: false,
    instagramConnected: false
  });
});

/* ================= FALLBACK ================= */

app.get("*", (req, res) => {
  if (req.path.startsWith("/api/")) {
    return res.status(404).json({
      ok: false,
      error: "API endpoint not found."
    });
  }

  res.sendFile(
    path.join(dir, "../public/index.html")
  );
});

/* ================= START ================= */

app.listen(port, () => {
  console.log("--------------------------------");
  console.log("NOVAAA AI V3.1.0");
  console.log(`Server running on port ${port}`);
  console.log(`Features listed: ${features.length}`);
  console.log(
    "Gemini model:",
    process.env.GEMINI_MODEL || "gemini-3.8-flash"
  );
  console.log("--------------------------------");
});
