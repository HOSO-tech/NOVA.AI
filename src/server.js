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

app.set("trust proxy", 1);
app.use(express.json({ limit: "2mb" }));

app.use(session({
  secret: process.env.SESSION_SECRET || "change-this-session-secret",
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    sameSite: "lax",
    secure: "auto",
    maxAge: 24 * 60 * 60 * 1000
  }
}));

app.use(express.static(path.join(dir, "../public")));

const baseUrl = () =>
  (process.env.APP_BASE_URL || `http://localhost:${port}`).replace(/\/$/, "");

/* ================= AI PROVIDERS ================= */

const providerReady = () => ({
  claude: !!process.env.ANTHROPIC_API_KEY,
  gemini: !!process.env.GEMINI_API_KEY,
  openai: !!process.env.OPENAI_API_KEY
});

async function askClaude(message, systemPrompt) {
  const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY
  });

  const response = await client.messages.create({
    model: process.env.CLAUDE_MODEL || "claude-3-5-sonnet-20241022",
    max_tokens: 2000,
    system: systemPrompt,
    messages: [{ role: "user", content: message }]
  });

  return response.content
    .filter(item => item.type === "text")
    .map(item => item.text)
    .join("\n");
}

async function askGemini(message, systemPrompt) {
  const client = new GoogleGenerativeAI(
    process.env.GEMINI_API_KEY
  );

  const model = client.getGenerativeModel({
    model: process.env.GEMINI_MODEL || "gemini-3.8-flash",
    systemInstruction: systemPrompt
  });

  const result = await model.generateContent(message);

  return result.response.text();
}

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

/* ================= FEATURE REGISTRY ================= */

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

/* ================= SYSTEM PROMPT ================= */

const mainPrompt = `
You are NOVAAA AI, an intelligent creative assistant.

You help users with:
- Stories and screenplays
- Horror, mystery and suspense
- Characters and world-building
- YouTube and Instagram content
- Video concepts and animation planning
- Dialogue, narration and voiceover
- Creative brainstorming and production planning

Be helpful, creative and clear.
Follow the user's requested language and format.
Do not claim to have performed actions that you have not performed.
`;

/* ================= HEALTH AND STATUS ================= */

app.get("/api/health", (_, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    server: "online",
    version: "3.0.1",
    timestamp: new Date().toISOString()
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    version: "3.0.1",
    server: "online",
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
      feature = ""
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

    const available = providerReady();

    if (!["claude", "gemini", "openai"].includes(provider)) {
      return res.status(400).json({
        ok: false,
        error: "Unknown AI provider."
      });
    }

    if (!available[provider]) {
      return res.status(400).json({
        ok: false,
        error: `${provider} is not configured. Add its API key in Render environment variables.`
      });
    }

    const instruction = [
      mainPrompt,
      feature ? `Current creative tool: ${feature}.` : "",
      systemPrompt
    ].filter(Boolean).join("\n\n");

    let answer;

    if (provider === "claude") {
      answer = await askClaude(message, instruction);
    } else if (provider === "gemini") {
      answer = await askGemini(message, instruction);
    } else {
      answer = await askOpenAI(message, instruction);
    }

    res.json({
      ok: true,
      provider,
      answer
    });

  } catch (error) {
    console.error("AI CHAT ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "AI request failed."
    });
  }
});

/* ================= CREATIVE COMMAND ROUTER ================= */

app.post("/api/command", async (req, res) => {
  try {
    const {
      command,
      input,
      provider = "gemini"
    } = req.body || {};

    if (!command || !input) {
      return res.status(400).json({
        ok: false,
        error: "A command and input are required."
      });
    }

    if (!["claude", "gemini", "openai"].includes(provider)) {
      return res.status(400).json({
        ok: false,
        error: "Unknown provider."
      });
    }

    if (!providerReady()[provider]) {
      return res.status(400).json({
        ok: false,
        error: `${provider} is not configured.`
      });
    }

    const prompt = `
You are NOVAAA AI's creative command system.

Perform this task: ${command}

User's input:
${input}

Give a complete, useful result.
Use headings when appropriate.
Do not include unnecessary introductions.
`;

    let answer;

    if (provider === "claude") {
      answer = await askClaude(prompt, mainPrompt);
    } else if (provider === "gemini") {
      answer = await askGemini(prompt, mainPrompt);
    } else {
      answer = await askOpenAI(prompt, mainPrompt);
    }

    res.json({
      ok: true,
      command,
      provider,
      answer
    });

  } catch (error) {
    console.error("COMMAND ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "Command failed."
    });
  }
});

/* ================= YOUTUBE OAUTH ================= */

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
      "YouTube OAuth is not configured. Check your Render environment variables."
    );
  }

  const state = crypto.randomBytes(24).toString("hex");

  req.session.youtubeOAuthState = state;

  try {
    await new Promise((resolve, reject) => {
      req.session.save(error => {
        if (error) reject(error);
        else resolve();
      });
    });

    const params = new URLSearchParams({
      client_id: process.env.GOOGLE_CLIENT_ID,
      redirect_uri: youtubeRedirectUri(),
      response_type: "code",
      scope: [
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/youtube.upload"
      ].join(" "),
      access_type: "offline",
      prompt: "consent",
      state
    });

    res.redirect(
      `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`
    );

  } catch (error) {
    console.error("YOUTUBE SESSION ERROR:", error);
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
      !req.session.youtubeOAuthState ||
      state !== req.session.youtubeOAuthState
    ) {
      return res.status(403).send(
        "Invalid OAuth state. Please try connecting again."
      );
    }

    delete req.session.youtubeOAuthState;

    if (!code) {
      return res.status(400).send(
        "Missing Google authorisation code."
      );
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
        "Google token exchange failed. Check your OAuth credentials."
      );
    }

    req.session.youtubeConnected = true;

    // Token persistence is not implemented yet.
    // YouTube publishing is not enabled by this connection alone.

    await new Promise((resolve, reject) => {
      req.session.save(error => {
        if (error) reject(error);
        else resolve();
      });
    });

    res.redirect("/?youtube=connected");

  } catch (error) {
    console.error("YOUTUBE CALLBACK ERROR:", error);
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

/* ================= INSTAGRAM STATUS ================= */

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
      ? "Instagram credentials are configured. OAuth flow still needs setup."
      : "Instagram credentials have not been configured."
  });
});

/* ================= PROJECT MEMORY ================= */

const projectMemory = new Map();

app.get("/api/projects", (req, res) => {
  const projects = projectMemory.get(req.sessionID) || [];

  res.json({
    ok: true,
    projects
  });
});

app.post("/api/projects", (req, res) => {
  const { name, content = "" } = req.body || {};

  if (!name || typeof name !== "string") {
    return res.status(400).json({
      ok: false,
      error: "Project name is required."
    });
  }

  const projects = projectMemory.get(req.sessionID) || [];

  const project = {
    id: crypto.randomUUID(),
    name,
    content,
    updatedAt: new Date().toISOString()
  };

  projects.push(project);
  projectMemory.set(req.sessionID, projects);

  res.json({
    ok: true,
    project
  });
});

app.delete("/api/projects/:id", (req, res) => {
  const projects = projectMemory.get(req.sessionID) || [];

  const updated = projects.filter(
    project => project.id !== req.params.id
  );

  projectMemory.set(req.sessionID, updated);

  res.json({
    ok: true,
    projects: updated
  });
});

/* ================= FALLBACK ================= */

app.get("*", (req, res, next) => {
  if (req.path.startsWith("/api/")) {
    return res.status(404).json({
      ok: false,
      error: "API endpoint not found."
    });
  }

  res.sendFile(path.join(dir, "../public/index.html"));
});

/* ================= START SERVER ================= */

app.listen(port, () => {
  console.log("--------------------------------");
  console.log("NOVAAA AI V3.0.1");
  console.log(`Server running on port ${port}`);
  console.log(`Features registered: ${features.length}`);
  console.log("--------------------------------");
});
