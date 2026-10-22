
## 1. Opening & context (3–5 min)

**Goal:** Establish who you are, why *you* care about orbitals + VR + intuition, and plant the “touch an orbital” question.

**Content:**

* Quick intro:

  * “I’m Audun, Chief Research Officer at Myreze; background in theoretical/quantum chemistry and chemistry education (Hylleraas center, KURT/UiO, learning assistant program).”
  * One sentence each on:

    * Myreze + VR / real-time 3D
    * 3D deep learning / generative models for geometry
    * Active learning / LA-program and why you care about pedagogy

* Hook:

  * Put the title question on screen: **“Is it possible to touch an orbital?”**
  * 1–2 images: a textbook orbital vs your VR orbital screenshot.
  * Brief callback to your 2022 blog post (you can show it but don’t explain it yet).

**Possible cuts:**
Don’t go into technical detail on Myreze’s pipelines here; just position them as “we know VR and real-time 3D very well, and I brought that mindset back to quantum chemistry.”

---

## 2. The “impossible” course & the intuition problem (5–7 min)

**Goal:** Ground the need for intuition in your personal teaching experience.

Use your **Physical Chemistry II – Intro to Quantum Chemistry** course as the narrative spine:

* Scope in one slide:

  * Intro QM (particle-in-a-box → hydrogen)
  * Many-body QM / quantum chemistry (helium, molecules, ab initio)
  * Time-dependent QM (IR, spectroscopy)

* Make the overload explicit:

  * “10 credits, but if you unpacked all topics, it’s basically a full bachelor.”
  * Point out the cognitive bottlenecks:

    * abstract notation (Hilbert spaces, bra-ket, determinants),
    * multiple scales (electrons ↔ molecules ↔ spectra),
    * weak visual intuition.

* Connect to active learning:

  * Mention the learning assistant program and how active learning helps, *but* even in LA-driven settings, students still struggle to **“see”** wavefunctions, correlation, orbital pictures beyond static blobs.

This sets up your thesis: *we need better embodied, interactive representations* if we want non-PhD students to really internalize these ideas.

**From your big Reveal deck:**
You can reuse 2–3 **“journey from big to small”** slides (e.g. the graded backgrounds, “en reise i skala”) to illustrate how you tried to get students to imagine scale and dynamics, *but* don’t go deep into history yet.

---

## 3. Quantum-literate workforce & the current gap (5–7 min)

**Goal:** Zoom out from your classroom to the global “quantum workforce” conversation.

You only need a **few sharp data points**, not a full literature review:

* **Workforce shortage & skill mismatch**

  * Greinert et al., *Future quantum workforce: competences, requirements and forecasts* (PRPER 2023) – industry and academic experts expect a **shortage of qualified people** and stress the need for both deep physics and engineering / software skills. ([arXiv][1])
  * Kaur & Venegas-Gomez, *Defining the quantum workforce landscape: a review of global quantum education initiatives* – they explicitly note the challenge of building a **diverse, sustainable quantum workforce** and map the current education landscape. ([arXiv][2])

* **Quantum education is still concentrated at grad/PhD level**

  * Perron et al., *Quantum Undergraduate Education and Scientific Training* – QIST education is **mostly at graduate/postdoc level**, with only early undergrad efforts, making it hard to prepare engineers and applied scientists in time. ([arXiv][3])

* **Students feel unprepared / unclear on where they fit**

  * Recent PER work surveying STEM undergrads shows they are aware of quantum as “the next big thing” but unsure what knowledge or courses they need, and many see it as “only for physics elites”. ([Physical Review Letters][4])

You can phrase it like:

> “Right now, most people working in quantum computing and high-end quantum chemistry come from highly academic backgrounds – physics masters and PhDs. The literature is pretty clear that engineering-level, technician-level, even ‘quantum-aware software developer’ skills are still scarce, and undergrad education is only starting to catch up.”

Then link back:

* “To get there, we need not just more courses, but better *intuition-building* tools. That’s exactly the problem I ran into in Physical Chemistry II.”

Optional add-on: one sentence on **ab initio + ML**:

* The demand for ab initio simulations in chemistry/materials (plus ML potentials) means more people must reason about **wavefunctions, basis sets, correlation, approximations** – not just run black-box DFT.

---

## 4. A compressed “first meeting with quantum chemistry” (7–8 min)

**Goal:** Give the audience just enough quantum chemistry structure so the VR part makes sense, without drowning them.

Use your existing Reveal.js deck, but **brutally condensed**.

### 4.1 From classical to wave mechanics (2–3 min)

Keep:

* One “universe is dynamic / structured” visual (e.g., galaxy-in-a-bottle + dandelion or fluid sim).
* Double-slit / De Broglie slides.
* Schrödinger equation + Born interpretation.

Message:

> “We moved students from particles and forces to wavefunctions and probabilities. That jump is already huge.”

### 4.2 From one particle to many (2–3 min)

Keep just a **minimal helium storyline**:

* Single slide with molecular Schrödinger equation for helium (not every derivation).
* One slide with the **variational principle**.
* One slide with **requirements on Ψ(x₁, x₂)** (normalization, antisymmetry, physically reasonable fragmentation).
* One slide introducing **Hartree product vs Slater determinant**.

Use these not as math lessons, but as **setup for what you want to visualize later**:

* “Where does electron correlation live?”
* “What does antisymmetry mean geometrically?”

### 4.3 Basis sets and orbitals (2–3 min)

Keep a **very small subset** of your basis-expansion slides:

* One “basis function” slide.
* One “basis expansion |ψ⟩ = Σ cᵢ |φᵢ⟩” slide.
* One slide showing local vs delocalized basis functions side-by-side.

Connect the dots explicitly:

> “In practice, we expand our many-electron wavefunctions in basis functions. Orbitals are one way we ‘package’ that structure. But students mostly see these as static isosurfaces in a book. What if we could *walk into* the basis expansion?”

**What to cut from this section:**

* Almost all of the historical atom model slides (Dalton, Rutherford, etc.) unless you have a *very* short story to tell with one composite slide.
* Detailed LJ potential sliders, fluid sims, Feynman quote, heavy CCSDT diagrams, PySCF demos — great for a longer quantum chem talk, but they’ll steal time from VR/orbitals.

---

## 5. Stepping inside: “Can you touch an orbital?” with Braketlab (10–12 min)

**Goal:** Pay off the title question and show VR + Braketlab as a concrete didactic tool.

### 5.1 The shrinking metaphor (2–3 min)

Reuse a few of your “reise i skala / krympe” slides:

* Sphere → world shrinking → “Hvordan er det å krympe?”
* Question: “Kan alt i naturen sanses?”

Then pivot:

> “VR lets us cheat. We can artificially enlarge, slow down, and externalize wavefunctions until they become navigable objects.”

### 5.2 Braketlab overview (3–4 min)

Very lightweight tech path:

* Jupyter + Braketlab: define model systems, orbitals, or approximate helium / H₂ wavefunctions.
* Export grids / isosurfaces as JSON/mesh.
* Load into web-based VR (WebXR, Evince, etc.) → headset.

Show:

* Screenshots or short video of:

  * A simple hydrogenic orbital (1s, 2p).
  * A molecular orbital (e.g. H₂O, as in your last slide).
  * Maybe a basic time-dependent example (wavepacket sloshing) if it doesn’t complicate things.

Key didactic points:

* Students can **walk around nodes**, slice through densities, “stick their head into” the orbital.
* They can change parameters in Python and *immediately* see the shape update – linking **computational thinking** to spatial intuition.

### 5.3 Quantum intuition lab: helium / correlation (4–5 min)

Here’s where you connect back to your helium setup:

* Start with a **Hartree product** approximation in Braketlab:

  * Show what the product of two 1s-like orbitals looks like; highlight that there’s no explicit electron–electron correlation aside from mean-field.

* Then a **Slater determinant**:

  * Explain visually what antisymmetry implies (nodal structure when electrons “swap”, exclusion).

* Then mention (or briefly show) a **Hylleraas-style correlated wavefunction**:

  * Even if you don’t render the full 6D structure, you can:

    * Show marginal densities,
    * Or show how the pair distribution changes, e.g., “correlation hole” around one electron.

The message:

> “Instead of telling students ‘there is electron correlation’, we can *show them* how different approximations change the shape of electron density and probability, in an environment their body can navigate.”

Tie back to **ab initio + ML**:

* “The same geometric intuitions – basis expansions, correlation holes, nodal surfaces – underpin modern ML potentials and quantum-chemistry pipelines used in materials discovery. VR can help students see and debug these models, not just run them.”

---

## 6. Closing the loop: from orbitals to workforce (3–5 min)

**Goal:** Land the plane and reconnect all threads: course overload, workforce needs, VR as a serious tool.

Key points:

1. *Answer the question*:

   > “Can you touch an orbital? Not literally, but we can get very close to a tactile, embodied understanding of wavefunctions.”

2. *Connect to workforce literature*:

   * Mention that frameworks like the **European Competence Framework for Quantum Technologies** explicitly call out a spectrum of roles and competences beyond academic theorists – including **software, engineering, and application-domain skills**. ([Frontiers][5])
   * VR + computational playgrounds like Braketlab are one way to make Hilbert space something that engineering students and applied scientists can actually reason about.

3. *Practical takeaway*:

   * Propose VR / interactive notebooks as:

     * lab modules in existing quantum courses,
     * on-ramp for non-physics students into quantum literacy,
     * potential bridge between **ab initio simulation** and **quantum computing** (e.g., visualizing state spaces, entanglement, basis changes).

4. End with a visual:

   * A final VR orbital shot with “Takk for meg / Thank you” and maybe a small QR code to the Braketlab docs and/or your blog post.

---

## Are there gaps / missing pieces?

A few things you might consider (if time allows):

* **Make the quantum computing connection explicit.**
  Even a single slide where you say:

  * “These orbitals live in a huge Hilbert space. Qubits are another way of slicing that space. The same need for intuition about superposition, basis choice, and entanglement appears in quantum computing.”
* **One explicit learning goal slide.**
  For example:

  * “After a VR orbital session, students should be able to:

    1. Verbally describe nodes and lobes in simple orbitals.
    2. Predict qualitatively how changing basis/approximation will change the shape.
    3. Connect these shapes to terms in a variational ansatz.”
       This keeps it grounded in pedagogy, not just cool tech.

If you *don’t* have systematic evaluation yet, you can frame it as:

> “I’ll share some preliminary observations and design ideas; formal evaluation is still future work.”

---

## What might be worth removing or shrinking?

Given the time, I’d strongly consider **cutting or minimising**:

* Detailed atomic-history sequence (Dalton → Rutherford → etc.) – keep at most one “puzzle of 1920” slide.
* Most of the LJ/fluids/Feynman material – it’s great for a thermodynamics/stat mech or forces talk, but off-focus here.
* Heavy CCSDT / multiparticle diagrams – they illustrate complexity, but risk losing the non-chemists in the room *right before* your VR payoff.

Those cuts buy you time to really **show** Braketlab and VR, which is what makes your talk unique in this workshop context.


[1]: https://arxiv.org/abs/2208.08249?utm_source=chatgpt.com "The Future Quantum Workforce: Competences, Requirements and Forecasts"
[2]: https://arxiv.org/pdf/2202.08940?utm_source=chatgpt.com "Deﬁning the quantum workforce landscape: a review of global quantum ..."
[3]: https://arxiv.org/abs/2109.13850?utm_source=chatgpt.com "Quantum Undergraduate Education and Scientific Training"
[4]: https://link.aps.org/doi/10.1103/PhysRevPhysEducRes.20.010138?utm_source=chatgpt.com "Science, technology, engineering, and mathematics undergraduates ..."
[5]: https://www.frontiersin.org/journals/quantum-science-and-technology/articles/10.3389/frqst.2023.1225733/full?utm_source=chatgpt.com "Towards a quantum ready workforce: the updated European Competence ..."
