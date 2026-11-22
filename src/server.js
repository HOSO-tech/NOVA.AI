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
const port = process.env.PORT || 3000;
const dir = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(dir, "..");

const generatedDir = path.join(root, "generated");
const videoDir = path.join(generatedDir, "videos");
const sceneDir = path.join(generatedDir, "scenes");

fs.mkdirSync(videoDir, { recursive: true });
fs.mkdirSync(sceneDir, { recursive: true });

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
app.use("/generated", express.static(generatedDir));

/* ================= SYSTEM ================= */

const mainPrompt = `
You are NOVAAA AI, an intelligent creative assistant.

You can help with:
- Stories
- Horror
- Mystery
- Screenplays
- Characters
- World building
- YouTube
- Instagram
- Video concepts
- Animation
- Creator tools
- Content production
- Creative brainstorming
- Video directing

Be creative, accurate and helpful.
Follow the user's requested language.
Never pretend an action happened when it did not.
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
  "System Health",
  "NOVAAA Video Director",
  "NOVAAA Scene Engine",
  "NOVAAA Video Renderer"
];

/* ================= PROVIDERS ================= */

const providerReady = () => ({
  gemini: !!process.env.GEMINI_API_KEY,
  claude: !!process.env.ANTHROPIC_API_KEY,
  openai: !!process.env.OPENAI_API_KEY
});

async function askGemini(message, systemPrompt) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("Gemini API key is missing.");
  }

  const client = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

  const model = client.getGenerativeModel({
    model: process.env.GEMINI_MODEL || "gemini-3.5-flash-lite",
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

async function askClaude(message, systemPrompt) {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("Claude API key is missing.");
  }

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

async function askOpenAI(message, systemPrompt) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key is missing.");
  }

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

/* ================= RESPONSE ================= */

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
    version: "4.0.0",
    timestamp: new Date().toISOString()
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    ok: true,
    name: "NOVAAA AI",
    server: "online",
    version: "4.0.0",
    providers: providerReady(),
    connections: {
      youtube: !!req.session.youtubeConnected,
      instagram: false
    },
    videoEngine: {
      enabled: true,
      renderer: "FFmpeg"
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
          const role =
            item?.role === "assistant"
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

/* ============================================================
   NOVAAA VIDEO ENGINE
   ============================================================ */

/* Create an AI video plan */

app.post("/api/video/plan", async (req, res) => {
  try {
    const {
      prompt,
      duration = 30,
      language = "English",
      style = "cinematic",
      scenes = 5
    } = req.body || {};

    if (!prompt) {
      return res.status(400).json({
        ok: false,
        error: "Video prompt is required."
      });
    }

    const sceneCount = Math.max(
      3,
      Math.min(Number(scenes) || 5, 12)
    );

    const planningPrompt = `
You are NOVAAA Video Director.

Create a video production plan.

User idea:
${prompt}

Language:
${language}

Visual style:
${style}

Target duration:
${duration} seconds

Create exactly ${sceneCount} scenes.

Return ONLY valid JSON in this format:

{
  "title": "video title",
  "description": "short description",
  "scenes": [
    {
      "number": 1,
      "duration": 5,
      "visual": "visual description",
      "camera": "camera direction",
      "narration": "narration",
      "sound": "sound design"
    }
  ]
}
`;

    const raw = await askGemini(
      planningPrompt,
      mainPrompt
    );

    let plan;

    try {
      const cleaned = raw
        .replace(/```json/gi, "")
        .replace(/```/g, "")
        .trim();

      plan = JSON.parse(cleaned);
    } catch {
      plan = {
        title: "NOVAAA Video",
        description: prompt,
        scenes: Array.from(
          { length: sceneCount },
          (_, index) => ({
            number: index + 1,
            duration: Math.max(
              2,
              Math.round(Number(duration) / sceneCount)
            ),
            visual: `${prompt} — scene ${index + 1}`,
            camera: "cinematic camera movement",
            narration: "",
            sound: "cinematic ambience"
          })
        )
      };
    }

    res.json({
      ok: true,
      plan
    });

  } catch (error) {
    console.error("VIDEO PLAN ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "Could not create video plan."
    });
  }
});

/* Create SVG scene cards */

function escapeXml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function wrapText(text, maxChars = 55) {
  const words = String(text || "").split(/\s+/);
  const lines = [];
  let current = "";

  for (const word of words) {
    if ((current + " " + word).trim().length > maxChars) {
      if (current) lines.push(current);
      current = word;
    } else {
      current = `${current} ${word}`.trim();
    }
  }

  if (current) lines.push(current);

  return lines.slice(0, 8);
}

function createSceneSVG(scene, totalScenes, title) {
  const visualLines = wrapText(scene.visual || "");
  const narrationLines = wrapText(scene.narration || "");

  const visualText = visualLines
    .map(
      (line, i) =>
        `<text x="80" y="${300 + i * 42}" font-size="30" fill="#eeeeee">${escapeXml(line)}</text>`
    )
    .join("");

  const narrationText = narrationLines
    .map(
      (line, i) =>
        `<text x="80" y="${660 + i * 34}" font-size="22" fill="#dddddd">${escapeXml(line)}</text>`
    )
    .join("");

  return `
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">
  <rect width="1280" height="720" fill="#16070d"/>
  <rect x="35" y="35" width="1210" height="650"
        rx="28" fill="#241018" stroke="#b9b9b9" stroke-width="3"/>

  <text x="80" y="105"
        font-family="Arial"
        font-size="28"
        fill="#c0c0c0">NOVAAA AI • VIDEO STUDIO</text>

  <text x="80" y="165"
        font-family="Arial"
        font-size="44"
        font-weight="bold"
        fill="#ffffff">${escapeXml(title)}</text>

  <text x="80" y="220"
        font-family="Arial"
        font-size="24"
        fill="#bdbdbd">SCENE ${scene.number} / ${totalScenes}</text>

  <line x1="80" y1="245" x2="1200" y2="245"
        stroke="#777777" stroke-width="2"/>

  ${visualText}

  <text x="80" y="620"
        font-family="Arial"
        font-size="20"
        fill="#aaaaaa">NARRATION</text>

  ${narrationText}

  <text x="1100" y="650"
        font-family="Arial"
        font-size="18"
        fill="#999999">${escapeXml(scene.duration)}s</text>
</svg>`;
}

/* Render actual MP4 */

function renderVideo(sceneFiles, outputFile, fps = 30) {
  return new Promise((resolve, reject) => {
    const listFile = path.join(
      videoDir,
      `${crypto.randomUUID()}.txt`
    );

    const listContent = sceneFiles
      .map(file => `file '${file.replace(/'/g, "'\\''")}'`)
      .join("\n");

    fs.writeFileSync(listFile, listContent);

    const args = [
      "-y",
      "-f", "concat",
      "-safe", "0",
      "-i", listFile,
      "-vf",
      "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
      "-r", String(fps),
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      outputFile
    ];

    const ffmpeg = spawn("ffmpeg", args);

    let errorText = "";

    ffmpeg.stderr.on("data", data => {
      errorText += data.toString();
    });

    ffmpeg.on("error", error => {
      try {
        fs.unlinkSync(listFile);
      } catch {}

      reject(
        new Error(
          `FFmpeg could not start. Make sure FFmpeg is installed. ${error.message}`
        )
      );
    });

    ffmpeg.on("close", code => {
      try {
        fs.unlinkSync(listFile);
      } catch {}

      if (code === 0) {
        resolve(outputFile);
      } else {
        reject(
          new Error(
            `FFmpeg failed: ${errorText.slice(-3000)}`
          )
        );
      }
    });
  });
}

app.post("/api/video/render", async (req, res) => {
  try {
    const { plan } = req.body || {};

    if (!plan || !Array.isArray(plan.scenes)) {
      return res.status(400).json({
        ok: false,
        error: "A valid video plan is required."
      });
    }

    const id = crypto.randomUUID();
    const title = plan.title || "NOVAAA Video";

    const sceneFiles = [];

    for (let i = 0; i < plan.scenes.length; i++) {
      const scene = plan.scenes[i];

      const svg = createSceneSVG(
        scene,
        plan.scenes.length,
        title
      );

      const file = path.join(
        sceneDir,
        `${id}-${i + 1}.svg`
      );

      fs.writeFileSync(file, svg);
      sceneFiles.push(file);
    }

    const output = path.join(
      videoDir,
      `${id}.mp4`
    );

    await renderVideo(sceneFiles, output);

    res.json({
      ok: true,
      message: "NOVAAA created the video.",
      video: `/generated/videos/${id}.mp4`,
      downloadUrl: `/generated/videos/${id}.mp4`,
      title
    });

  } catch (error) {
    console.error("VIDEO RENDER ERROR:", error);

    res.status(500).json({
      ok: false,
      error: error.message || "Video rendering failed."
    });
  }
});

/* One-click video */

app.post("/api/video/create", async (req, res) => {
  try {
    const {
      prompt,
      duration = 30,
      language = "English",
      style = "cinematic"
    } = req.body || {};

    if (!prompt) {
      return res.status(400).json({
        ok: false,
        error: "Enter a video idea."
      });
    }

    const sceneCount = Math.max(
      3,
      Math.min(8, Math.ceil(Number(duration) / 6))
    );

    const planningPrompt = `
You are NOVAAA AI Video Director.

Create a ${duration}-second ${style} video.

Idea:
${prompt}

Language:
${language}

Create exactly ${sceneCount} scenes.

Return ONLY JSON:

{
  "title": "title",
  "scenes": [
    {
      "number": 1,
      "duration": 5,
      "visual": "visual description",
      "camera": "camera movement",
      "narration": "short narration",
      "sound": "sound"
    }
  ]
}
`;

    const raw = await askGemini(
      planningPrompt,
      mainPrompt
    );

    let plan;

    try {
      plan = JSON.parse(
        raw
          .replace(/```json/gi, "")
          .replace(/```/g, "")
          .trim()
      );
    } catch {
      plan = {
        title: "NOVAAA AI Video",
        scenes: Array.from(
          { length: sceneCount },
          (_, i) => ({
            number: i + 1,
            duration: Math.round(
              Number(duration) / sceneCount
            ),
            visual: `${prompt} — cinematic scene ${i + 1}`,
            camera: "slow cinematic movement",
            narration: "",
            sound: "cinematic ambience"
          })
        )
      };
    }

    const id = crypto.randomUUID();
    const sceneFiles = [];

    for (let i = 0; i < plan.scenes.length; i++) {
      const file = path.join(
        sceneDir,
        `${id}-${i + 1}.svg`
      );

      fs.writeFileSync(
        file,
        createSceneSVG(
          plan.scenes[i],
          plan.scenes.length,
          plan.title || "NOVAAA AI Video"
        )
      );

      sceneFiles.push(file);
    }

    const output = path.join(
      videoDir,
      `${id}.mp4`
    );

    await renderVideo(sceneFiles, output);

    res.json({
      ok: true,
      title: plan.title,
      plan,
      video: `/generated/videos/${id}.mp4`,
      downloadUrl: `/generated/videos/${id}.mp4`
    });

  } catch (error) {
    console.error("ONE CLICK VIDEO E
