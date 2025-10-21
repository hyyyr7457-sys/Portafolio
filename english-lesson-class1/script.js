"use strict";

// ---------- Data ----------
const PROFESSIONS = [
  { word: "lawyer", emoji: "⚖️", clue: "I work in a court.", article: "a" },
  { word: "engineer", emoji: "🧑‍💻", clue: "I design and build things.", article: "an" },
  { word: "doctor", emoji: "🩺", clue: "I help sick people at a hospital.", article: "a" },
  { word: "teacher", emoji: "👩‍🏫", clue: "I work at a school.", article: "a" },
  { word: "nurse", emoji: "👩‍⚕️", clue: "I assist patients and doctors.", article: "a" },
  { word: "police officer", emoji: "👮", clue: "I keep people safe.", article: "a" },
  { word: "chef", emoji: "👨‍🍳", clue: "I cook in a restaurant.", article: "a" },
  { word: "firefighter", emoji: "🚒", clue: "I fight fires.", article: "a" },
  { word: "artist", emoji: "🎨", clue: "I paint and draw.", article: "an" },
  { word: "farmer", emoji: "👩‍🌾", clue: "I work on a farm.", article: "a" },
  { word: "pilot", emoji: "✈️", clue: "I fly planes.", article: "a" },
  { word: "driver", emoji: "🚐", clue: "I drive a vehicle.", article: "a" }
];

const ADJECTIVES = ["friendly", "honest", "hardworking", "serious", "creative"];

const FIND_SOMEONE_WHO = [
  "works at night",
  "wears a uniform",
  "works outdoors",
  "speaks two languages",
  "likes creative work",
  "is very friendly",
  "works with computers",
  "helps people every day",
  "works in a team",
  "travels for work",
  "studies English three times a week",
  "attends English classes on Tuesday, Thursday and Friday"
];

// ---------- TTS (Web Speech API) ----------
let availableVoices = [];
let selectedVoice = null;

function loadVoices() {
  availableVoices = window.speechSynthesis.getVoices();
  // Prefer male en-US if possible
  const maleEnUs = availableVoices.find(v => v.lang && v.lang.startsWith("en-US") && /male|david|joey|mike|john/i.test(v.name));
  const anyEnUs = availableVoices.find(v => v.lang && v.lang.startsWith("en-US"));
  selectedVoice = maleEnUs || anyEnUs || availableVoices[0] || null;
  populateVoiceSelect();
}

function populateVoiceSelect() {
  const select = document.getElementById("voiceSelect");
  if (!select) return;
  select.innerHTML = "";
  for (const v of availableVoices) {
    const opt = document.createElement("option");
    opt.value = v.name;
    opt.textContent = `${v.name} — ${v.lang}`;
    if (selectedVoice && v.name === selectedVoice.name) opt.selected = true;
    select.appendChild(opt);
  }
}

function setVoiceByName(name) {
  const found = availableVoices.find(v => v.name === name);
  if (found) selectedVoice = found;
}

function speak(text, { rate = 1, pitch = 1 } = {}) {
  if (!text) return;
  const u = new SpeechSynthesisUtterance(text);
  if (selectedVoice) u.voice = selectedVoice;
  u.lang = selectedVoice?.lang || "en-US";
  u.rate = rate;
  u.pitch = pitch;
  window.speechSynthesis.speak(u);
}

async function speakSequence(items, gapMs = 450) {
  for (const item of items) {
    await waitForVoicesToFinish();
    speak(item);
    await sleep(gapMs);
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function waitForVoicesToFinish() {
  return new Promise(resolve => {
    const iv = setInterval(() => {
      if (!window.speechSynthesis.speaking) {
        clearInterval(iv);
        resolve();
      }
    }, 50);
  });
}

// ---------- Warm-up ----------
function buildGuessGrid() {
  const grid = document.getElementById("guessGrid");
  grid.innerHTML = "";
  PROFESSIONS.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = "card";

    const emoji = document.createElement("div");
    emoji.className = "emoji";
    emoji.textContent = p.emoji;

    const title = document.createElement("div");
    title.className = "title";
    title.textContent = "Guess the profession";

    const clue = document.createElement("div");
    clue.className = "clue";
    clue.textContent = p.clue;

    const answer = document.createElement("div");
    answer.className = "answer hidden";
    answer.textContent = p.word;

    const controls = document.createElement("div");
    controls.className = "controls";

    const btnClue = document.createElement("button");
    btnClue.className = "btn";
    btnClue.type = "button";
    btnClue.textContent = "🔊 Pista";
    btnClue.addEventListener("click", () => speak(p.clue));

    const btnReveal = document.createElement("button");
    btnReveal.className = "btn";
    btnReveal.type = "button";
    btnReveal.textContent = "👀 Revelar";
    btnReveal.addEventListener("click", () => {
      answer.classList.remove("hidden");
      answer.classList.add("reveal");
      speak(`You're ${p.article} ${p.word}!`);
    });

    controls.append(btnClue, btnReveal);
    card.append(emoji, title, clue, answer, controls);
    grid.appendChild(card);
  });
}

// ---------- Grammar Builder ----------
function fillDatalists() {
  const profsDatalist = document.getElementById("professionsList");
  const adjsDatalist = document.getElementById("adjectivesList");
  profsDatalist.innerHTML = PROFESSIONS.map(p => `<option value="${p.word}"></option>`).join("");
  adjsDatalist.innerHTML = ADJECTIVES.map(a => `<option value="${a}"></option>`).join("");
}

function currentBeForm(pronoun) {
  switch (pronoun) {
    case "I": return "am";
    case "You": return "are";
    case "He":
    case "She": return "is";
    case "We":
    case "They": return "are";
    default: return "am";
  }
}

function buildSentence() {
  const pronoun = document.getElementById("pronounSelect").value || "I";
  const profession = (document.getElementById("professionInput").value || "").trim();
  const adjective = (document.getElementById("adjectiveInput").value || "").trim();
  const be = currentBeForm(pronoun);
  const article = /^[aeiou]/i.test(profession) ? "an" : "a";

  const hasProfession = profession.length > 0;
  const hasAdjective = adjective.length > 0;

  let sentence = pronoun + " " + be + "";
  if (hasProfession) sentence += ` ${article} ${profession}`;
  if (hasAdjective) sentence += (hasProfession ? ". " + capitalize(pronoun) + " " + be + " " + adjective + "." : " " + adjective + ".");
  if (!hasProfession && !hasAdjective) sentence = `${pronoun} ${be}.`;

  const out = document.getElementById("sentenceOutput");
  out.textContent = sentence;
  document.getElementById("speakSentence").disabled = sentence.trim().length === 0;
}

function capitalize(s){ return s.charAt(0).toUpperCase() + s.slice(1); }

// ---------- Vocabulary Cards ----------
function buildVocabGrid() {
  const grid = document.getElementById("vocabGrid");
  grid.innerHTML = "";
  PROFESSIONS.forEach(p => {
    const card = document.createElement("div");
    card.className = "card vocab-card";

    const header = document.createElement("div");
    header.className = "inline-actions";
    const emoji = document.createElement("div");
    emoji.className = "emoji";
    emoji.textContent = p.emoji;

    const word = document.createElement("div");
    word.className = "word";
    word.textContent = p.word;

    header.append(emoji, word);

    const spelling = document.createElement("div");
    spelling.className = "spelling";
    spelling.textContent = p.word.split("").join(" · ");

    const controls = document.createElement("div");
    controls.className = "controls";

    const btnListen = document.createElement("button");
    btnListen.className = "btn";
    btnListen.type = "button";
    btnListen.textContent = "▶️ Listen";
    btnListen.addEventListener("click", () => speak(p.word));

    const btnEcho = document.createElement("button");
    btnEcho.className = "btn";
    btnEcho.type = "button";
    btnEcho.textContent = "🔁 Echo";
    btnEcho.addEventListener("click", async () => {
      await speakSequence([p.word, p.word], 650);
    });

    const btnSpell = document.createElement("button");
    btnSpell.className = "btn";
    btnSpell.type = "button";
    btnSpell.textContent = "🔤 Spell";
    btnSpell.addEventListener("click", async () => {
      const letters = p.word.toUpperCase().split("");
      await speakSequence(letters, 180);
    });

    controls.append(btnListen, btnEcho, btnSpell);
    card.append(header, spelling, controls);
    grid.appendChild(card);
  });
}

function playAllOnce(){ speakSequence(PROFESSIONS.map(p=>p.word), 500); }
function playAllEcho(){
  const seq = PROFESSIONS.flatMap(p => [p.word, p.word]);
  speakSequence(seq, 450);
}

// ---------- Role play ----------
function randomFrom(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

function generateRoleplay(){
  const p1 = randomFrom(PROFESSIONS);
  const p2 = randomFrom(PROFESSIONS);
  const s = [
    `A: What do you do?`,
    `B: I'm ${p1.article} ${p1.word}. I work ${p1.clue.replace(/^I /, '').toLowerCase()}`,
    `A: Nice!`,
    `B: What do you do?`,
    `A: I'm ${p2.article} ${p2.word}. ${capitalize(p2.clue)}`
  ].join("\n");
  const out = document.getElementById("rpOutput");
  out.textContent = s;
  speakSequence([
    `What do you do?`,
    `I'm ${p1.article} ${p1.word}.`,
    p1.clue,
    `What do you do?`,
    `I'm ${p2.article} ${p2.word}.`,
    p2.clue
  ], 650);
}

// ---------- Find Someone Who ----------
function buildFSW(){
  const ul = document.getElementById("fswList");
  ul.innerHTML = "";
  FIND_SOMEONE_WHO.forEach(t => {
    const li = document.createElement("li");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    const label = document.createElement("label");
    label.textContent = `Find someone who ${t}.`;
    li.append(cb, label);
    ul.appendChild(li);
  });
}

// ---------- Describe notes ----------
function wireDescribeSpeakers(){
  document.querySelectorAll(".describe-item .speak-note").forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.parentElement.querySelector("textarea").value.trim();
      if (text) speak(text);
    });
  });
}

// ---------- Homework tools ----------
function insertRandomSentence(){
  const pronouns = ["I", "You", "He", "She", "We", "They"];
  const pronoun = randomFrom(pronouns);
  const p = randomFrom(PROFESSIONS);
  const a = randomFrom(ADJECTIVES);
  const be = currentBeForm(pronoun);
  const sentence = `${pronoun} ${be} ${p.article} ${p.word}. ${capitalize(pronoun)} ${be} ${a}.`;
  const ta = document.getElementById("hwTextarea");
  const prefix = ta.value.trim().length ? "\n" : "";
  ta.value += prefix + sentence;
  updateHWCount();
}

function updateHWCount(){
  const ta = document.getElementById("hwTextarea");
  const lines = ta.value.split(/\n+/).filter(l => l.trim().length);
  document.getElementById("hwCount").textContent = `${lines.length} / 10 oraciones`;
}

function downloadHW(){
  const text = document.getElementById("hwTextarea").value;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "homework_class1.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------- Voice UI ----------
function wireVoiceUI(){
  const select = document.getElementById("voiceSelect");
  select.addEventListener("change", (e) => setVoiceByName(e.target.value));
  document.getElementById("testVoice").addEventListener("click", () => speak("Hello! This is the voice for our class."));
}

// ---------- Init ----------
function init(){
  // Build UI
  buildGuessGrid();
  buildVocabGrid();
  buildFSW();
  fillDatalists();
  wireDescribeSpeakers();
  wireVoiceUI();

  // Grammar builder interactions
  document.getElementById("buildSentence").addEventListener("click", buildSentence);
  document.getElementById("pronounSelect").addEventListener("change", buildSentence);
  document.getElementById("professionInput").addEventListener("input", buildSentence);
  document.getElementById("adjectiveInput").addEventListener("input", buildSentence);
  document.getElementById("speakSentence").addEventListener("click", () => {
    const text = document.getElementById("sentenceOutput").textContent;
    if (text) speak(text);
  });

  // Vocab global buttons
  document.getElementById("playAllOnce").addEventListener("click", playAllOnce);
  document.getElementById("playAllEcho").addEventListener("click", playAllEcho);

  // Role play
  document.getElementById("rpGenerate").addEventListener("click", generateRoleplay);

  // Homework
  document.getElementById("insertRandom").addEventListener("click", insertRandomSentence);
  document.getElementById("downloadHW").addEventListener("click", downloadHW);
  document.getElementById("hwTextarea").addEventListener("input", updateHWCount);
  updateHWCount();

  // Voices
  if (typeof speechSynthesis !== "undefined"){
    loadVoices();
    speechSynthesis.onvoiceschanged = loadVoices;
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
