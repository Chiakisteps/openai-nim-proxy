// server.js - OpenAI to NVIDIA NIM API Proxy (2026対応版)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 推論プロセス表示 (true = <think>タグで表示, false = 最終回答のみ)
const SHOW_REASONING = false;

// 🔥 シンキングモード (一部モデルで思考有効化)
const ENABLE_THINKING_MODE = false;

// ===== 2026年5月時点のモデルマッピング =====
// ⚠️ build.nvidia.com/models で最新ラインナップを確認してください
// ⚠️ 廃止済み: kimi-k2-instruct, glm-4.7, gemma-3-27b など
const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'meta/llama-3.3-70b-instruct',        // 安定・高速
  'gpt-4':          'minimax/minimax-m2.7',                 // コーディング強力
  'gpt-4-turbo':    'qwen/qwen3-coder-480b-a35b-instruct', // 256kコンテキスト
  'gpt-4o':         'z-ai/glm-5.1', // 人気No.1
  'claude-3-opus':  'mistralai/mistral-large-3-675b-instruct', // 汎用最強
  'claude-3-sonnet': 'nvidia/nemotron-3-super-120b-v1',    // NVIDIA製
  'gemini-pro':     'google/gemma-4-31b-it',               // Googleの最新
};

// ヘルスチェック
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    version: '2026.05',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    rate_limit_info: '40 RPM (default) / 200 RPM (with application)'
  });
});

// モデル一覧 (OpenAI互換)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// チャット補完エンドポイント (メインプロキシ)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // モデル選択 (フォールバック付き)
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      // モデル名を直接試す
      try {
        const testRes = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: s => s < 500
        });
        if (testRes.status >= 200 && testRes.status < 300) nimModel = model;
      } catch (e) { /* ignore */ }

      // それでもなければ能力ベースでフォールバック
      if (!nimModel) {
        const ml = model.toLowerCase();
        if (ml.includes('gpt-4') || ml.includes('opus') || ml.includes('405b')) {
          nimModel = 'mistralai/mistral-large-3-675b-instruct';
        } else if (ml.includes('claude') || ml.includes('gemini') || ml.includes('70b')) {
          nimModel = 'meta/llama-3.3-70b-instruct';
        } else {
          nimModel = 'meta/llama-4-maverick-17b-128e-instruct';
        }
      }
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false,
      ...(ENABLE_THINKING_MODE && {
        extra_body: { chat_template_kwargs: { thinking: true } }
      })
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buf = '';
      let thinkStarted = false;

      response.data.on('data', chunk => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { res.write(line + '\n'); return; }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;
              let combined = '';

              if (SHOW_REASONING) {
                if (reasoning && !thinkStarted) {
                  combined = '<think>\n' + reasoning;
                  thinkStarted = true;
                } else if (reasoning) {
                  combined = reasoning;
                }
                if (content && thinkStarted) {
                  combined += '\n</think>\n\n' + content;
                  thinkStarted = false;
                } else if (content) {
                  combined += content;
                }
                delta.content = combined || '';
              } else {
                delta.content = content || '';
              }
              delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => { console.error('Stream error:', err); res.end(); });

    } else {
      const choices = response.data.choices.map(choice => {
        let content = choice.message?.content || '';
        if (SHOW_REASONING && choice.message?.reasoning_content) {
          content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
        }
        return {
          index: choice.index,
          message: { role: choice.message.role, content },
          finish_reason: choice.finish_reason
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices,
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    // レート制限エラー(429)の場合は分かりやすいメッセージを返す
    if (error.response?.status === 429) {
      return res.status(429).json({
        error: {
          message: 'Rate limit exceeded (40 RPM). Please wait and retry.',
          type: 'rate_limit_error',
          code: 429
        }
      });
    }
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `${req.path} not found`, type: 'not_found', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`[NIM Proxy 2026] Running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
});
