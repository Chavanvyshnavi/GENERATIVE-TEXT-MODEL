# task4_textgen.py
# Task 4: Generative Text Model
# - Tries an LSTM using numpy if available
# - Falls back to a simple Markov chain generator if numpy is missing
# Usage: python task4_textgen.py
# Author: (you)

import sys
import random
import math
import textwrap

# ========== Training corpus (multi-topic) ==========
CORPUS = """
SUSTAINABILITY AND ETHICS:
Sustainable design balances function, cost, and environmental impact.
Cardboard reuse shows circular materials can be elegant, durable, and expressive.
Ethical sourcing requires traceability, fair labor, and transparent reporting.
Consumers reward brands that minimize waste and invest in local repair networks.

AI AND MACHINE LEARNING:
A perceptron separates linearly separable data with a single hyperplane.
When data are not linearly separable, deeper architectures with nonlinearities are required.
Temperature controls randomness in text sampling: low values are conservative; high values are creative.
Responsible AI emphasizes data governance, measurable robustness, and human oversight.

BIOINFORMATICS AND METABOLOMICS:
Metabolomic profiles capture snapshots of cellular chemistry in response to environment and disease.
Feature normalization, batch correction, and robust cross-validation reduce false discoveries.
Pathway enrichment links molecular signals to biological interpretation.
Reproducible analysis depends on versioned code, metadata, and careful documentation.

PROJECT MANAGEMENT:
Clear scope, visible milestones, and honest risk logs keep complex projects on track.
Buffers absorb uncertainty; retrospectives convert mistakes into systems for learning.
Communication plans align teams, while lightweight dashboards prevent status surprise.
Continuous improvement relies on small, testable changes delivered frequently.
""".strip()

# ========== Utilities ==========
def pretty_print_title(s):
    print("\n" + "="*len(s))
    print(s)
    print("="*len(s))

def join_sentences(text):
    # Clean up whitespace and ensure sentences are wrapped
    text = " ".join(text.split())
    return "\n\n".join(textwrap.wrap(text, width=80))

# ========== OPTION A: Try LSTM with NumPy ==========
USE_LSTM = False
try:
    import numpy as np
    USE_LSTM = True
except Exception:
    USE_LSTM = False

if USE_LSTM:
    pretty_print_title("NUMPY DETECTED — RUNNING MINI LSTM (CHAR-LEVEL)")
    # ---- Simple, small char-level LSTM (keeps memory small and iterations low) ----
    # Note: This is intentionally tiny so it runs in limited environments.
    rng = np.random.default_rng(1)

    data = CORPUS
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}

    # hyperparams
    hidden_size = 64
    seq_length = 80
    learning_rate = 0.1
    n_iter = 600  # increase if you have time

    # initialize params
    def randn(*shape):
        return rng.normal(0, 1.0/np.sqrt(sum(shape)/2), size=shape).astype(np.float32)

    params = {
        'W_x': randn(vocab_size, 4*hidden_size),
        'W_h': randn(hidden_size, 4*hidden_size),
        'b'  : np.zeros((1, 4*hidden_size), dtype=np.float32),
        'W_y': randn(hidden_size, vocab_size),
        'b_y': np.zeros((1, vocab_size), dtype=np.float32),
    }
    mem = {k: np.zeros_like(v) for k, v in params.items()}

    # helpers
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    def softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def one_hot(ix):
        v = np.zeros((1, vocab_size), dtype=np.float32)
        v[0, ix] = 1.0
        return v

    def lstm_step(x, h, c):
        W_x, W_h, b = params['W_x'], params['W_h'], params['b']
        z = x @ W_x + h @ W_h + b
        H = h.shape[1]
        i = sigmoid(z[:, 0:H])
        f = sigmoid(z[:, H:2*H])
        o = sigmoid(z[:, 2*H:3*H])
        g = np.tanh(z[:, 3*H:4*H])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        cache = (x, h, c, i, f, o, g, c_new)
        return h_new, c_new, cache

    def forward_backward(inputs, targets, hprev, cprev):
        xs, hs, cs, ys, ps, caches = {}, {}, {}, {}, {}, {}
        hs[-1] = hprev
        cs[-1] = cprev
        loss = 0.0
        for t in range(len(inputs)):
            xs[t] = one_hot(inputs[t])
            hs[t], cs[t], caches[t] = lstm_step(xs[t], hs[t-1], cs[t-1])
            ys[t] = hs[t] @ params['W_y'] + params['b_y']
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][0, targets[t]] + 1e-12)

        dparams = {k: np.zeros_like(v) for k, v in params.items()}
        dh_next = np.zeros_like(hs[0])
        dc_next = np.zeros_like(cs[0])

        H = hprev.shape[1]
        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[0, targets[t]] -= 1.0
            dparams['W_y'] += hs[t].T @ dy
            dparams['b_y'] += dy

            dh = dy @ params['W_y'].T + dh_next
            x, h_prev, c_prev, i, f, o, g, c_t = caches[t]

            do = dh * np.tanh(c_t)
            dct = dh * o * (1 - np.tanh(c_t)**2) + dc_next

            di = dct * g
            dg = dct * i
            df = dct * c_prev
            dc_prev = dct * f

            di_input = di * i * (1 - i)
            df_input = df * f * (1 - f)
            do_input = do * o * (1 - o)
            dg_input = dg * (1 - g**2)

            dz = np.hstack([di_input, df_input, do_input, dg_input])

            dparams['W_x'] += x.T @ dz
            dparams['W_h'] += h_prev.T @ dz
            dparams['b']  += dz

            dh_next = dz @ params['W_h'].T
            dc_next = dc_prev

        for k in dparams:
            np.clip(dparams[k], -5, 5, out=dparams[k])
        return loss, dparams, hs[len(inputs)-1], cs[len(inputs)-1]

    def adagrad_update(dparams):
        for k in params:
            mem[k] += dparams[k] * dparams[k]
            params[k] += -learning_rate * dparams[k] / (np.sqrt(mem[k]) + 1e-8)

    def sample(seed_ix, n, temperature=0.8, h=None, c=None):
        if h is None:
            h = np.zeros((1, hidden_size), dtype=np.float32)
        if c is None:
            c = np.zeros((1, hidden_size), dtype=np.float32)
        x = one_hot(seed_ix)
        ixes = [seed_ix]
        for t in range(n):
            h, c, _ = lstm_step(x, h, c)
            y = h @ params['W_y'] + params['b_y']
            p = softmax(y / max(temperature, 1e-6))
            ix = int(np.argmax(rng.multinomial(1, p.flatten())))
            x = one_hot(ix)
            ixes.append(ix)
        return ''.join(ix_to_char[i] for i in ixes)

    # Prepare training indices
    data_ix = [char_to_ix[ch] for ch in data]
    p = 0
    hprev = np.zeros((1, hidden_size), dtype=np.float32)
    cprev = np.zeros((1, hidden_size), dtype=np.float32)
    smooth_loss = -math.log(1.0/vocab_size) * seq_length

    # Training loop (small demo)
    for it in range(1, n_iter + 1):
        if p + seq_length + 1 >= len(data_ix):
            hprev = np.zeros_like(hprev)
            cprev = np.zeros_like(cprev)
            p = 0
        inputs = data_ix[p:p+seq_length]
        targets = data_ix[p+1:p+seq_length+1]
        loss, dparams, hprev, cprev = forward_backward(inputs, targets, hprev, cprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        adagrad_update(dparams)
        p += seq_length
        if it % 150 == 0 or it == 1:
            print(f"iter {it}, loss {smooth_loss:.2f}")
            # sample a short sequence
            seed = random.choice(inputs)
            print("sample:", sample(seed, 200, temperature=0.9)[:400])
            print("----")

    def generate_paragraph(topic, temperature=0.9, length=450):
        prompt = f"Topic: {topic}.\\n\\nParagraph: "
        # feed prompt to LSTM to set state
        h = np.zeros((1, hidden_size), dtype=np.float32)
        c = np.zeros((1, hidden_size), dtype=np.float32)
        generated = ""
        for ch in prompt:
            ix = char_to_ix.get(ch, char_to_ix.get(' ',0))
            h, c, _ = lstm_step(one_hot(ix), h, c)
            generated += ch
        sampled = sample(char_to_ix.get(prompt[-1], 0), length, temperature, h=h, c=c)
        raw = generated + sampled
        # tidy
        raw = " ".join(raw.split())
        # cut to a sentence end if possible
        end = max(raw.rfind('. '), raw.rfind('! '), raw.rfind('? '))
        if end > 150:
            raw = raw[:end+1]
        return raw

    # Demo generate
    print("\n=== GENERATED PARAGRAPH (LSTM) ===\n")
    print(generate_paragraph("project management"))
    sys.exit(0)

# ========== OPTION B: Markov chain fallback (no numpy) ==========
pretty_print_title("NUMPY NOT AVAILABLE — RUNNING MARKOV CHAIN FALLBACK")

# Word-level Markov chain trainer / sampler
def build_markov(corpus, k=2):
    words = []
    for line in corpus.splitlines():
        line = line.strip()
        if line:
            # split simple punctuation
            for token in line.replace(":", " : ").split():
                words.append(token)
    model = {}
    for i in range(len(words) - k):
        key = tuple(words[i:i+k])
        nxt = words[i+k]
        model.setdefault(key, []).append(nxt)
    return model, words

def generate_markov_paragraph(model, words, topic, k=2, max_words=120):
    # seed: if topic words exist in corpus pick nearby, else choose random key
    topic_tokens = topic.split()
    seed_key = None
    for i in range(len(words)-k):
        key = tuple(words[i:i+k])
        if any(tok.lower() in (w.lower() for w in key) for tok in topic_tokens):
            seed_key = key
            break
    if seed_key is None:
        seed_key = random.choice(list(model.keys()))
    out = list(seed_key)
    for _ in range(max_words - k):
        key = tuple(out[-k:])
        choices = model.get(key)
        if not choices:
            break
        out.append(random.choice(choices))
    # join tokens into a readable paragraph
    text = " ".join(out)
    # basic cleanup: fix spaces before punctuation
    text = text.replace(" : ", ": ")
    # ensure sentences end with a period
    if text[-1] not in ".!?":
        text = text.rstrip(". ,") + "."
    # postwrap for readability
    return " ".join(text.split())

# Build model and demo
markov_model, markov_words = build_markov(CORPUS, k=2)
print("\n=== GENERATED PARAGRAPH (Markov fallback) ===\n")
print(generate_markov_paragraph(markov_model, markov_words, "project management", k=2, max_words=120))

# Small interactive prompt for user
def interactive_prompt():
    print("\nYou can now request paragraphs on any topic.")
    print("Type a topic and press Enter (or just press Enter to quit).")
    while True:
        topic = input("\nTopic> ").strip()
        if not topic:
            print("Goodbye.")
            break
        if USE_LSTM:
            # This branch won't be reached because we sys.exit after LSTM demo,
            # but place here if someone modifies the script.
            print(generate_paragraph(topic))
        else:
            print("\n" + generate_markov_paragraph(markov_model, markov_words, topic, k=2, max_words=140))

if __name__ == "__main__":
    # run interactive only in fallback mode
    if not USE_LSTM:
        interactive_prompt()


