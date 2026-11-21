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
const app=express(), port=process.env.PORT||3000, dir=path.dirname(fileURLToPath(import.meta.url));
app.use(express.json({limit:"1mb"}));
app.use(session({
 secret: process.env.SESSION_SECRET || "development-only-change-this",
 resave: false, saveUninitialized: false,
 cookie: { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", maxAge: 24*60*60*1000 }
}));
app.use(express.static(path.join(dir,"../public")));

const baseUrl = () => process.env.APP_BASE_URL || `http://localhost:${port}`;
const oauthReady = {
 youtube: !!(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET),
 instagram: !!(process.env.INSTAGRAM_APP_ID && process.env.INSTAGRAM_APP_SECRET)
};
app.get("/api/social/status", (req,res) => res.json({
 configured: oauthReady,
 connected: { youtube: !!req.session.youtube, instagram: !!req.session.instagram },
 accounts: {
  youtube: req.session.youtube?.channelTitle || null,
  instagram: req.session.instagram?.username || null
 }
}));
function beginOAuth(req,res,provider,authorizeUrl,params){
 if(!oauthReady[provider]) return res.status(503).send(`${provider} connection is not configured on the server yet.`);
 const state=crypto.randomBytes(24).toString("hex");
 req.session.oauthState={provider,state};
 const u=new URL(authorizeUrl);
 for(const [k,v] of Object.entries({...params,state})) u.searchParams.set(k,v);
 res.redirect(u.toString());
}
app.get("/auth/youtube", (req,res)=>beginOAuth(req,res,"youtube","https://accounts.google.com/o/oauth2/v2/auth",{
 client_id:process.env.GOOGLE_CLIENT_ID,
 redirect_uri:`${baseUrl()}/auth/youtube/callback`,
 response_type:"code", access_type:"offline", prompt:"consent",
 scope:"openid email https://www.googleapis.com/auth/youtube.readonly"
}));
app.get("/auth/youtube/callback",async(req,res)=>{
 try{
  if(!req.session.oauthState || req.session.oauthState.provider!=="youtube" || req.query.state!==req.session.oauthState.state) return res.status(400).send("Invalid OAuth state. Return to NOVAAA and try again.");
  req.session.oauthState=null;
  if(req.query.error) return res.status(400).send("YouTube authorization was not completed.");
  const tokenRes=await fetch("https://oauth2.googleapis.com/token",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body:new URLSearchParams({
   code:req.query.code,client_id:process.env.GOOGLE_CLIENT_ID,client_secret:process.env.GOOGLE_CLIENT_SECRET,
   redirect_uri:`${baseUrl()}/auth/youtube/callback`,grant_type:"authorization_code"
  })});
  const token=await tokenRes.json(); if(!token.access_token) throw new Error("Token exchange failed");
  const ch=await fetch("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",{headers:{Authorization:`Bearer ${token.access_token}`}});
  const data=await ch.json(); if(!ch.ok) throw new Error("Could not read YouTube channel");
  req.session.youtube={accessToken:token.access_token,refreshToken:token.refresh_token||null,channelTitle:data.items?.[0]?.snippet?.title||"YouTube channel"};
  res.redirect("/?connected=youtube");
 }catch(e){console.error("YouTube OAuth:",e.message);res.status(502).send("YouTube connection failed. Check the OAuth setup and redirect URL.");}
});
app.get("/auth/instagram",(req,res)=>beginOAuth(req,res,"instagram","https://www.instagram.com/oauth/authorize",{
 client_id:process.env.INSTAGRAM_APP_ID,
 redirect_uri:`${baseUrl()}/auth/instagram/callback`,
 response_type:"code",scope:"instagram_business_basic"
}));
app.get("/auth/instagram/callback",async(req,res)=>{
 try{
  if(!req.session.oauthState || req.session.oauthState.provider!=="instagram" || req.query.state!==req.session.oauthState.state) return res.status(400).send("Invalid OAuth state. Return to NOVAAA and try again.");
  req.session.oauthState=null;
  if(req.query.error) return res.status(400).send("Instagram authorization was not completed.");
  const tokenRes=await fetch("https://api.instagram.com/oauth/access_token",{method:"POST",body:new URLSearchParams({
   client_id:process.env.INSTAGRAM_APP_ID,client_secret:process.env.INSTAGRAM_APP_SECRET,
   grant_type:"authorization_code",redirect_uri:`${baseUrl()}/auth/instagram/callback`,code:req.query.code
  })});
  const token=await tokenRes.json(); if(!token.access_token) throw new Error("Token exchange failed");
  const profileRes=await fetch(`https://graph.instagram.com/me?fields=user_id,username&access_token=${encodeURIComponent(token.access_token)}`);
  const profile=await profileRes.json(); if(!profileRes.ok) throw new Error("Could not read Instagram profile");
  req.session.instagram={accessToken:token.access_token,userId:profile.user_id,username:profile.username||"Instagram account"};
  res.redirect("/?connected=instagram");
 }catch(e){console.error("Instagram OAuth:",e.message);res.status(502).send("Instagram connection failed. Check Meta app setup, account eligibility and redirect URL.");}
});
app.post("/api/social/disconnect/:provider",(req,res)=>{
 if(!["youtube","instagram"].includes(req.params.provider)) return res.status(400).json({error:"Unknown provider"});
 delete req.session[req.params.provider]; res.json({ok:true});
});
const keys={claude:!!process.env.ANTHROPIC_API_KEY,gemini:!!process.env.GEMINI_API_KEY,openai:!!process.env.OPENAI_API_KEY};
app.get("/api/status",(_,r)=>r.json({providers:keys,server:"online"}));
app.post("/api/chat",async(req,res)=>{
 const {provider,message,history=[]}=req.body||{};
 if(!["claude","gemini","openai"].includes(provider))return res.status(400).json({error:"Choose a provider."});
 if(!keys[provider])return res.status(503).json({error:`${provider} is not connected. Add its key in hosting environment variables.`});
 if(typeof message!=="string"||!message.trim()||message.length>12000)return res.status(400).json({error:"Message is empty or too long."});
 const h=Array.isArray(history)?history.slice(-10).filter(x=>x&&["user","assistant"].includes(x.role)&&typeof x.content==="string"):[];
 try{
  let reply="";
  if(provider==="claude"){
   const c=new Anthropic({apiKey:process.env.ANTHROPIC_API_KEY});
   const out=await c.messages.create({model:"claude-3-5-haiku-latest",max_tokens:1400,system:"You are NOVAAA AI, a helpful and safe creative assistant.",messages:[...h,{role:"user",content:message}]});
   reply=out.content.filter(x=>x.type==="text").map(x=>x.text).join("\n");
  }else if(provider==="gemini"){
   const g=new GoogleGenerativeAI(process.env.GEMINI_API_KEY).getGenerativeModel({model:"gemini-1.5-flash"});
   const transcript=h.map(x=>`${x.role}: ${x.content}`).join("\n");
   reply=(await g.generateContent(`${transcript}\nuser: ${message}`)).response.text();
  }else{
   const c=new OpenAI({apiKey:process.env.OPENAI_API_KEY});
   const out=await c.chat.completions.create({model:"gpt-4o-mini",max_tokens:1400,messages:[{role:"system",content:"You are NOVAAA AI, a helpful and safe creative assistant."},...h,{role:"user",content:message}]});
   reply=out.choices[0]?.message?.content||"No response returned.";
  }
  res.json({reply});
 }catch(e){console.error(e.message);res.status(502).json({error:"Provider request failed. Check key, model access and usage limits."});}
});
app.get("*",(_,r)=>r.sendFile(path.join(dir,"../public/index.html")));
app.listen(port,()=>console.log(`NOVAAA AI on ${port}`));
