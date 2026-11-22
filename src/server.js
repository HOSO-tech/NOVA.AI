import express from "express";
import dotenv from "dotenv";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from "openai";
import session from "express-session";
import crypto from "node:crypto";
import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 3000;
const dir = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(dir, "..");

const generatedDir = path.join(root, "generated");
const videoDir = path.join(generatedDir, "videos");
const sceneDir = path.join(generatedDir, "scenes");
const dataDir = process.env.DATA_DIR || path.join(root, "data");
const projectsFile = path.join(dataDir, "projects.json");

for (const folder of [generatedDir, videoDir, sceneDir, dataDir]) {
  fs.mkdirSync(folder, { recursive: true });
}

const baseUrl = () =>
  (process.env.APP_BASE_URL || `http://localhost:${port}`).replace(/\/$/, "");

app.set("trust proxy", 1);
app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true, limit: "2mb" }));

app.use(
  session({
    name: "novaaa.sid",
    secret:
      process.env.SESSION_SECRET ||
      "change-this-session-secret-before-production",
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      maxAge: 24 * 60 * 60 * 1000
    }
  })
);

app.use(express.static(path.join(dir, "../public")));
app.use("/generated", express.static(generatedDir));

/* -------------------- SYSTEM -------------------- */

const mainPrompt = `
You are NOVAAA AI, an intelligent creative assistant.

You help with:
- Stories, horror, mystery and screenplays
- Characters, dialogue and world building
- YouTube and Instagram content planning
- Video concepts, animation and video directing
- Creator tools, production workflows and brainstorming

Be creative, accurate and useful. Follow the user's requested language.
Never claim that an action was completed unless the server actually completed it.
`;

const features = [
  "AI Chat", "Story Generator", "Script Writer", "Horror Story Builder",
  "Dialogue Generator", "Character Creator", "Character Backstory",
  "Plot Generator", "Plot Twist Generator", "Scene Generator",
  "Episode Planner", "Series Planner", "YouTube Ideas", "YouTube Titles",
  "YouTube Descriptions", "YouTube Tags", "YouTube Hooks", "Shorts Ideas",
  "Instagram Ideas", "Instagram Captions", "Creator Prompts", "Image Prompts",
  "Video Prompts", "Voiceover Scripts", "Thumbnail Ideas", "Content Planner",
  "Project Memory", "Project Notes", "Creative Brainstorm", "Rewrite Assistant",
  "Tone Changer", "Dialogue Enhancer", "Story Expander", "Story Shortener",
  "Story Continuation", "Ending Generator", "Opening Hook Generator",
  "Suspense Builder", "Mystery Builder", "Horror Builder", "Comedy Builder",
  "Action Builder", "Fantasy Builder", "Sci-Fi Builder", "Drama Builder",
  "Animation Planner", "Animatic Planner", "Shot List", "Scene Breakdown",
  "Camera Direction", "Lighting Direction", "Sound Design", "Music Direction",
  "Voice Direction", "Character Voice Guide", "Location Generator",
  "Prop Generator", "Costume Generator", "World Builder", "Lore Generator",
  "Episode Recap", "Episode Preview", "Trailer Script", "Teaser Script",
  "Cliffhanger Generator", "Comment Reply Ideas", "Community Post Ideas",
  "Poll Ideas", "Trend Brainstorm", "Content Calendar", "Upload Checklist",
  "Creator Checklist", "Production Checklist", "SEO Assistant", "Keyword Ideas",
  "Hashtag Ideas", "Brand Name Generator", "Channel Name Generator",
  "Series Name Generator", "Character Name Generator", "Video Concept Generator",
  "Short Film Generator", "Film Treatment", "Logline Generator",
  "Synopsis Generator", "Pitch Generator", "Scene Dialogue",
  "Monologue Generator", "Narration Generator", "First Person Rewrite",
  "Third Person Rewrite", "Hindi Script", "English Script", "Hinglish Script",
  "Translation Assistant", "Grammar Assistant", "Summarizer", "Idea Organizer",
  "Project Dashboard", "YouTube Connection", "Instagram Connection",
  "Social Status", "Provider Status", "Command Centre", "NOVAAA Assistant",
  "Creative Director", "Production Assistant", "Content Assistant",
  "Project Assistant", "Feature Router", "Workspace Manager", "Memory Manager",
  "Safety Filter", "System Health", "NOVAAA Video Director",
  "NOVAAA Scene Engine", "NOVAAA Video Renderer"
];

/* -------------------- PROVIDERS -------------------- */

const providerReady = () => ({
  gemini: Boolean(process.env.GEMINI_API_KEY),
  claude: Boolean(process.env.ANTHROPIC_API_KEY),
  openai: Boolean(process.env.OPENAI_API_KEY)
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function askGemini(message, systemPrompt = mainPrompt) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("Gemini API key is missing. Add GEMINI_API_KEY in Render environment variables.");
  }

  const client = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const model = client.getGenerativeModel({
    model: process.env.GEMINI_MODEL || "gemini-2.0-flash",
    systemInstruction: systemPrompt
  });

  let lastError;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const result = await model.generateContent(message);
      return result.response.text();
    } catch (error) {
      lastError = error;
      const temporary = /503|429|500|502|504|overload|high demand|temporarily unavailable/i
        .test(String(error?.message || error));
      if (!temporary || attempt === 2) break;
      await sleep(1000 * (2 ** attempt));
    }
  }
  throw lastError || new Error("Gemini request failed.");
}

async function askClaude(message, systemPrompt = mainPrompt) {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("Claude API key is missing. Add ANTHROPIC_API_KEY in Render environment variables.");
  }

  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const response = await client.messages.create({
    model: process.env.CLAUDE_MODEL || "claude-3-5-sonnet-20241022",
    max_tokens: 2500,
    system: systemPrompt,
    messages: [{ role: "user", content: message }]
  });

  return (response.content || [])
    .filter((item) => item.type === "text")
    .map((item) => item.text)
    .join("\n");
}

async function askOpenAI(message, systemPrompt = mainPrompt) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key is missing. Add OPENAI_API_KEY in Render environment variables.");
  }

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const response = await client.chat.completions.create({
    model: process.env.OPENAI_MODEL || "gpt-4o-mini",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: message }
    ]
  });

  return response.choices?.[0]?.message?.content || "";
}

async function askProvider(provider, message, systemPrompt = mainPrompt) {
  if (provider === "gemini") return askGemini(message, systemPrompt);
  if (provider === "claude") return askClaude(message, systemPrompt);
  if (provider === "openai") return askOpenAI(message, systemPrompt);
  throw new Error("Unknown AI provider. Choose gemini, claude or openai.");
}

/* -------------------- RESPONSE HELPERS -------------------- */

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

function cleanText(value, max = 12000) {
  return String(value ?? "").trim().slice(0, max);
}

function asyncHandler(handler) {
  return (req, res, next) =>
    Promise.resolve(handler(req, res, next)).catch(next);
}

/* -------------------- PROJECT MEMORY STORAGE -------------------- */

function readProjects() {
  try {
    if (!fs.existsSync(projectsFile)) return [];
    const parsed = JSON.parse(fs.readFileSync(projectsFile, "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error("PROJECT FILE READ ERROR:", error);
    return [];
  }
}

function writeProjects(projects) {
  const temporaryFile = `${projectsFile}.tmp`;
  fs.writeFileSync(temporaryFile, JSON.stringify(projects, null, 2), "utf8");
  fs.renameSync(temporaryFile, projectsFile);
}

function listProjects(res) {
  const projects = readProjects().sort(
    (a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt)
  );
  return res.json({ ok: true, projects, memories: projects });
}

function createProject(req, res) {
  const title = cleanText(req.body?.title, 160);
  const content = cleanText(req.body?.content, 20000);

  if (!title || !content) {
    return res.status(400).json({
      ok: false,
      error: "Both title and content are required."
    });
  }

  const projects = readProjects();
  const now = new Date().toISOString();
  const project = {
    id: crypto.randomUUID(),
    title,
    content,
    createdAt: now,
    updatedAt: now
  };

  projects.unshift(project);
  writeProjects(projects.slice(0, 250));

  return res.status(201).json({
    ok: true,
    project,
    projects: readProjects()
  });
}

app.get("/api/projects", (req, res) => listProjects(res));
app.post("/api/projects", (req, res) => createProject(req, res));
app.get("/api/memory", (req, res) => listProjects(res));
app.post("/api/memory", (req, res) => createProject(req, res));

app.delete("/api/projects/:id", (req, res) => {
  const projects = readProjects();
  const remaining = projects.filter((project) => project.id !== req.params.id);
  if (remaining.length === projects.length) {
    return res.status(404).json({ ok: false, error: "Project not found." });
  }
  writeProjects(remaining);
  return res.json({ ok: true, projects: remaining });
});

/* -------------------- STATUS -------------------- */

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    server: "online",
    version: "5.0.0",
    timestamp: new Date().toISOString()
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    server: "online",
    version: "5.0.0",
    providers: providerReady(),
    connections: {
      youtube: Boolean(req.session.youtubeConnected),
      instagram: Boolean(req.session.instagramConnected)
    },
    youtube: Boolean(req.session.youtubeConnected),
    instagram: Boolean(req.session.instagramConnected),
    videoEngine: {
      enabled: true,
      renderer: "FFmpeg scene-card renderer",
      generatesAIFootage: false
    },
    featureCount: features.length,
    projectCount: readProjects().length
  });
});

app.get("/api/social/status", (req, res) => {
  res.json({
    ok: true,
    youtube: Boolean(req.session.youtubeConnected),
    youtubeConnected: Boolean(req.session.youtubeConnected),
    instagram: Boolean(req.session.instagramConnected),
    instagramConnected: Boolean(req.session.instagramConnected),
    note: "Social OAuth and publishing must be configured separately."
  });
});

app.get("/api/providers", (req, res) => {
  res.json({ ok: true, providers: providerReady() });
});

app.get("/api/features", (req, res) => {
  res.json({ ok: true, count: features.length, features });
});

/* -------------------- AI CHAT -------------------- */

app.post("/api/chat", asyncHandler(async (req, res) => {
  const {
    message,
    provider = "gemini",
    systemPrompt = "",
    feature = "",
    history = []
  } = req.body || {};

  if (typeof message !== "string" || !message.trim()) {
    return res.status(400).json({ ok: false, error: "Please enter a message." });
  }
  if (message.length > 15000) {
    return res.status(400).json({ ok: false, error: "Message is too long." });
  }

  const instruction = [mainPrompt, feature ? `Current tool: ${cleanText(feature, 200)}` : "", cleanText(systemPrompt, 5000)]
    .filter(Boolean)
    .join("\n\n");

  const recent = Array.isArray(history)
    ? history.slice(-8).map((item) => {
        const role = item?.role === "assistant" ? "Assistant" : "User";
        return `${role}: ${cleanText(item?.content, 3000)}`;
      }).join("\n")
    : "";

  const prompt = recent
    ? `Recent conversation:\n${recent}\n\nUser:\n${message.trim()}`
    : message.trim();

  const answer = await askProvider(provider, prompt, instruction);
  return sendAIResult(res, provider, answer);
}));

/* -------------------- COMMAND CENTRE -------------------- */

app.post("/api/command", asyncHandler(async (req, res) => {
  const {
    command,
    input = "",
    prompt = "",
    provider = "gemini",
    feature = ""
  } = req.body || {};

  const task = cleanText(command || prompt, 12000);
  const userInput = cleanText(input, 12000);

  if (!task && !userInput) {
    return res.status(400).json({ ok: false, error: "Enter a command or prompt." });
  }

  const fullPrompt = `
NOVAAA CREATIVE TASK: ${cleanText(feature, 200) || task}

Task:
${task}

User input:
${userInput}

Complete the task directly. Use headings where useful.
Follow the user's requested language.
`;

  const answer = await askProvider(provider, fullPrompt, mainPrompt);
  return sendAIResult(res, provider, answer, { command: task });
}));

/* -------------------- VIDEO PLANNING -------------------- */

function extractJson(text) {
  const cleaned = String(text || "")
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const first = cleaned.indexOf("{");
    const last = cleaned.lastIndexOf("}");
    if (first >= 0 && last > first) {
      return JSON.parse(cleaned.slice(first, last + 1));
    }
    throw new Error("The AI provider did not return valid video-plan JSON.");
  }
}

async function buildVideoPlan({
  prompt,
  duration = 30,
  language = "English",
  style = "cinematic",
  scenes = 5
}) {
  const sceneCount = Math.max(3, Math.min(Number(scenes) || 5, 12));
  const totalDuration = Math.max(6, Math.min(Number(duration) || 30, 180));

  const planningPrompt = `
You are NOVAAA Video Director.
Create a concise production plan for the user's video idea.

User idea: ${cleanText(prompt, 5000)}
Language: ${cleanText(language, 80)}
Visual style: ${cleanText(style, 120)}
Target duration: ${totalDuration} seconds
Create exactly ${sceneCount} scenes.

Return ONLY valid JSON:
{
  "title": "video title",
  "description": "short description",
  "scenes": [
    {
      "number": 1,
      "duration": 5,
      "visual": "visual description",
      "camera": "camera direction",
      "narration": "short narration",
      "sound": "sound design"
    }
  ]
}
`;

  let plan;
  try {
    plan = extractJson(await askGemini(planningPrompt, mainPrompt));
  } catch (error) {
    // A fallback plan is provided only for a planning response-format failure.
    // Provider/API errors still surface to the user.
    if (/API key is missing|quota|permission|unauthorized|not found|404|403|401/i.test(error.message)) {
      throw error;
    }
    const perScene = Math.max(2, Math.floor(totalDuration / sceneCount));
    plan = {
      title: "NOVAAA Video",
      description: cleanText(prompt, 1000),
      scenes: Array.from({ length: sceneCount }, (_, index) => ({
        number: index + 1,
        duration: perScene,
        visual: `${cleanText(prompt, 400)} — scene ${index + 1}`,
        camera: "Cinematic wide shot",
        narration: "",
        sound: "Atmospheric background sound"
      }))
    };
  }

  if (!plan || !Array.isArray(plan.scenes) || !plan.scenes.length) {
    throw new Error("The video planner returned an empty scene list.");
  }

  plan.title = cleanText(plan.title || "NOVAAA Video", 160);
  plan.description = cleanText(plan.description || prompt, 1000);
  plan.scenes = plan.scenes.slice(0, 12).map((scene, index) => ({
    number: index + 1,
    duration: Math.max(2, Math.min(Number(scene?.duration) || 4, 60)),
    visual: cleanText(scene?.visual, 1200),
    camera: cleanText(scene?.camera, 400),
    narration: cleanText(scene?.narration, 1000),
    sound: cleanText(scene?.sound, 400)
  }));

  return plan;
}

app.post("/api/video/plan", asyncHandler(async (req, res) => {
  const prompt = cleanText(req.body?.prompt, 5000);
  if (!prompt) {
    return res.status(400).json({ ok: false, error: "Video prompt is required." });
  }

  const plan = await buildVideoPlan({
    prompt,
    duration: req.body?.duration,
    language: req.body?.language,
    style: req.body?.style,
    scenes: req.body?.scenes
  });

  return res.json({ ok: true, plan });
}));

/* -------------------- VIDEO RENDERING -------------------- */

/*
  This renderer creates a text-based storyboard/slideshow MP4 from the plan.
  It does NOT generate AI footage, character animation, or voice audio.
  For true generative video, a separate video-generation provider is required.
*/

function escapeFilterPath(filePath) {
  return filePath
    .replace(/\\/g, "\\\\\\\\")
    .replace(/:/g, "\\:")
    .replace(/'/g, "\\'");
}

function wrapText(value, maxChars = 58) {
  const words = cleanText(value, 1600).split(/\s+/);
  const lines = [];
  let line = "";

  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (candidate.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 8).join("\n");
}

function writeTextFile(id, name, value) {
  const file = path.join(sceneDir, `${id}-${name}.txt`);
  fs.writeFileSync(file, cleanText(value, 4000), "utf8");
  return escapeFilterPath(file);
}

function renderVideo(plan) {
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    const outputFile = path.join(videoDir, `${id}.mp4`);
    const scenes = plan.scenes.slice(0, 12);
    const args = ["-y"];
    const filters = [];
    const labels = [];
    let totalDuration = 0;

    scenes.forEach((scene, index) => {
      const duration = Math.max(2, Math.min(Number(scene.duration) || 4, 60));
      totalDuration += duration;

      args.push(
        "-f", "lavfi",
        "-i", `color=c=0x16070d:s=1280x720:r=30:d=${duration}`
      );

      const titleFile = writeTextFile(id, `title-${index}`, plan.title);
      const visualFile = writeTextFile(id, `visual-${index}`, wrapText(scene.visual));
      const narrationFile = writeTextFile(id, `narration-${index}`, wrapText(scene.narration || " "));
      const fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
      const fontOption = fs.existsSync(fontPath)
        ? `fontfile=${escapeFilterPath(fontPath)}:`
        : "";

      const filter = [
        `[${index}:v]`,
        `drawtext=${fontOption}textfile='${titleFile}':fontcolor=0xD6D6D6:fontsize=28:x=80:y=75:box=1:boxcolor=0x241018@0.8:boxborderw=14`,
        `drawtext=${fontOption}textfile='${visualFile}':fontcolor=white:fontsize=38:line_spacing=18:x=80:y=220:box=1:boxcolor=0x241018@0.75:boxborderw=22`,
        `drawtext=${fontOption}text='SCENE ${index + 1} / ${scenes.length}':fontcolor=0xC0C0C0:fontsize=22:x=80:y=160`,
        `drawtext=${fontOption}textfile='${narrationFile}':fontcolor=0xDDDDDD:fontsize=22:line_spacing=10:x=80:y=570:box=1:boxcolor=0x241018@0.8:boxborderw=12`,
        `drawtext=${fontOption}text='${duration}s':fontcolor=0xC0C0C0:fontsize=20:x=1120:y=650`,
        `format=yuv420p[v${index}]`
      ].join(",");

      filters.push(filter);
      labels.push(`[v${index}]`);
    });

    filters.push(`${labels.join("")}concat=n=${scenes.length}:v=1:a=0[outv]`);

    args.push(
      "-filter_complex", filters.join(";"),
      "-map", "[outv]",
      "-an",
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-r", "30",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      "-t", String(Math.min(totalDuration, 180)),
      outputFile
    );

    const ffmpeg = spawn("ffmpeg", args);
    let errorText = "";

    ffmpeg.stderr.on("data", (chunk) => {
      errorText = (errorText + chunk.toString()).slice(-8000);
    });

    ffmpeg.on("error", (error) => {
      reject(new Error(`FFmpeg could not start. Confirm FFmpeg is installed on the hosting service. ${error.message}`));
    });

    ffmpeg.on("close", (code) => {
      if (code === 0 && fs.existsSync(outputFile)) {
        resolve({
          id,
          outputFile,
          url: `/generated/videos/${id}.mp4`,
          duration: totalDuration
        });
      } else {
        reject(new Error(`FFmpeg rendering failed. ${errorText.slice(-2500)}`));
      }
    });
  });
}

app.post("/api/video/render", asyncHandler(async (req, res) => {
  const plan = req.body?.plan;
  if (!plan || !Array.isArray(plan.scenes) || plan.scenes.length < 1) {
    return res.status(400).json({ ok: false, error: "A valid video plan is required." });
  }
  if (plan.scenes.length > 12) {
    return res.status(400).json({ ok: false, error: "A maximum of 12 scenes is supported." });
  }

  const result = await renderVideo({
    title: cleanText(plan.title || "NOVAAA Video", 160),
    scenes: plan.scenes.map((scene) => ({
      duration: scene.duration,
      visual: cleanText(scene.visual, 1200),
      narration: cleanText(scene.narration, 1000)
    }))
  });

  return res.json({
    ok: true,
    message: "NOVAAA rendered a storyboard video.",
    video: result.url,
    downloadUrl: result.url,
    title: cleanText(plan.title || "NOVAAA Video", 160),
    duration: result.duration,
    note: "This is a scene-card slideshow, not AI-generated footage or voice audio."
  });
}));

app.post("/api/video/create", asyncHandler(async (req, res) => {
  const prompt = cleanText(req.body?.prompt, 5000);
  if (!prompt) {
    return res.status(400).json({ ok: false, error: "Enter a video idea first." });
  }

  const plan = await buildVideoPlan({
    prompt,
    duration: req.body?.duration,
    language: req.body?.language,
    style: req.body?.style,
    scenes: req.body?.scenes || 5
  });

  const result = await renderVideo(plan);

  return res.json({
    ok: true,
    message: "NOVAAA rendered your scene-card video.",
    plan,
    video: result.url,
    downloadUrl: result.url,
    title: plan.title,
    duration: result.duration,
    note: "This is a scene-card slideshow. It does not contain AI-generated footage or voice audio."
  });
}));

/* -------------------- FALLBACKS & ERRORS -------------------- */

app.get("/api/*", (req, res) => {
  res.status(404).json({
    ok: false,
    error: `API route not found: ${req.path}`
  });
});

app.use((error, req, res, next) => {
  console.error("NOVAAA SERVER ERROR:", error);
  if (res.headersSent) return next(error);
  res.status(500).json({
    ok: false,
    error: error?.message || "An unexpected server error occurred."
  });
});

app.get("*", (req, res, next) => {
  if (req.path.startsWith("/generated/")) return next();
  res.sendFile(path.join(dir, "../public/index.html"));
});

app.listen(port, "0.0.0.0", () => {
  console.log("======================================");
  console.log(" NOVAAA AI V5.0.0");
  console.log(` Server running on port ${port}`);
  console.log(` Providers: ${JSON.stringify(providerReady())}`);
  console.log(` Features listed: ${features.length}`);
  console.log(" Project Memory: ENABLED (JSON file)");
  console.log(" Video Engine: FFmpeg scene-card renderer");
  console.log("======================================");
});
