/* Отчёт: один рендер на оба экрана.

   Жюри должно увидеть отчёт в момент `finish`, а не скачать файл, — поэтому
   он рисуется и у методиста, и у сотрудника из одного и того же кода. Двух
   шаблонов — под собеседование и под тренировку — здесь нет намеренно:
   механика везде одна, критерии и цитаты, меняется только шапка.

   Цитата ведёт в транскрипт. Разница принципиальная: «коммуникация 3 из 5» —
   мнение модели, «3 из 5, вот реплика, где кандидат поплыл» — инструмент. */

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const clock = ts => ts
  ? new Date(ts * 1000).toLocaleTimeString('ru-RU', {hour12: false})
  : '';

function duration(sec) {
  if (sec === null || sec === undefined) return '—';
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? `${m} мин ${s} с` : `${s} с`;
}

function personaLine(p) {
  if (!p || !p.role) return '';
  const bits = [];
  if (p.tone) bits.push(`тон: ${p.tone}`);
  if (p.strictness) bits.push(`строгость: ${p.strictness}`);
  if (p.pressure) bits.push(`давление: ${p.pressure}`);
  if (p.start_emotion) bits.push(`стартовая эмоция: ${p.start_emotion}`);
  return `<div class="rep-persona"><b>Собеседник.</b> ${esc(p.role)}` +
         (bits.length ? `. ${esc(bits.join('; '))}` : '') + `</div>`;
}

/* Кто на какую реплику сослался: обратный индекс, чтобы в транскрипте было
   видно, что именно эта реплика чему-то стоила баллов. */
function citationIndex(rep) {
  const byTurn = new Map();
  for (const c of rep.criteria || []) {
    for (const q of c.citations || []) {
      if (q.turn === undefined || q.turn < 0) continue;
      if (!byTurn.has(q.turn)) byTurn.set(q.turn, []);
      byTurn.get(q.turn).push({title: c.title, score: q.score});
    }
  }
  return byTurn;
}

export function renderReport(root, rep, opts = {}) {
  const ns = opts.ns || 'rep';
  const turnId = i => `${ns}-turn-${i}`;
  const cited = citationIndex(rep);
  const scored = (rep.criteria || []).filter(c => c.score !== null);

  const facts = [
    ['этапов', `${rep.stages_reached}/${rep.stages_total}`],
    ['реплик', rep.user_turns],
    ['длительность', duration(rep.duration_s)],
    ['покрытие критериев', `${Math.round((rep.coverage || 0) * 100)}%`],
    ['критериев с цитатой', `${rep.cited || 0}/${(rep.criteria || []).length}`],
  ];

  const head = `
    <div class="rep-head">
      <div class="rep-title">${esc(rep.scenario_title || 'Тренировка')}</div>
      <div class="rep-sub">${esc(rep.scenario_type || '')} ·
        ${esc(clock(rep.created_at))} ·
        ${rep.completed ? 'диалог завершён' : 'диалог идёт'}
        ${rep.finish_reason ? '(' + esc(rep.finish_reason) + ')' : ''}</div>
      <div class="rep-facts">
        <span>итог <b class="rep-overall">${rep.overall ?? '—'}</b></span>
        ${facts.map(([k, v]) => `<span>${k} <b>${esc(v)}</b></span>`).join('')}
      </div>
      ${personaLine(rep.persona)}
    </div>`;

  const conclusion = rep.conclusion
    ? `<h2>Вывод</h2><div class="rep-conclusion">${esc(rep.conclusion)}</div>`
    : (rep.completed ? '' : '<p class="rep-note">Общий вывод появится, ' +
                            'когда диалог завершится.</p>');

  const rows = (rep.criteria || []).map(c => {
    const quotes = (c.citations || []).filter(q => q.rationale || q.turn >= 0);
    const chips = quotes.length ? `<div class="cites">` + quotes.map(q => {
      const live = q.turn >= 0;
      const label = live ? `реплика ${q.turn + 1}` : 'без ссылки';
      return `<span class="cite${live ? '' : ' dead'}"
                    ${live ? `data-turn="${q.turn}"` : ''}
                    title="${esc(q.rationale)}">${label}` +
             (q.score !== null && q.score !== undefined
               ? ` · <b>${q.score}</b>` : '') + `</span>`;
    }).join('') + `</div>` : '';
    const why = c.rationale || (c.observations || []).join('; ');
    return `<tr>
      <td>${esc(c.title)}<div class="anchors">${esc(c.key)}</div></td>
      <td class="score${c.score === null ? ' none' : ''}">${c.score ?? 'нет оценки'}</td>
      <td>${esc(why) || '<span class="rep-empty">по этому критерию ' +
                        'в разговоре ничего не проявилось</span>'}${chips}</td>
    </tr>`;
  }).join('');

  const table = `<h2>Оценка по критериям</h2>
    <table><thead><tr>
      <th style="width:26%">критерий</th><th style="width:10%">оценка</th>
      <th>обоснование и цитаты</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="3" class="rep-empty">критериев нет</td></tr>'}</tbody></table>
    ${scored.length ? '' : '<p class="rep-note">Фоновая оценка ещё не ' +
      'накопила баллов — она дописывает их после каждой реплики.</p>'}`;

  const line = (t, i) => {
    const who = t.role === 'agent' ? 'Собеседник' : 'Сотрудник';
    const marks = cited.get(i) || [];
    const conf = t.typing && t.typing.confidence &&
                 t.typing.confidence !== 'нет данных' ? t.typing.confidence : '';
    return `<div class="line ${t.role}" id="${turnId(i)}">
      <div class="meta"><span class="who">${who}</span>${esc(clock(t.at))}</div>
      <div>
        <div class="text">${esc(t.text)}${t.interrupted
          ? '<span class="tag">перебит</span>' : ''}${conf
          ? `<span class="tag">${esc(conf)}</span>` : ''}</div>
        ${marks.length ? `<div class="cited-by">↑ ${marks.map(m =>
          esc(m.title) + (m.score === null || m.score === undefined
            ? '' : ` ${m.score}`)).join(' · ')}</div>` : ''}
      </div>
    </div>`;
  };

  const lines = (rep.transcript || []).map(line).join('') ||
    '<div class="line"><div class="meta"></div>' +
    '<div class="rep-empty">разговор ещё не начался</div></div>';
  const transcript = `<h2>Ход разговора</h2>
    <div class="rep-transcript">${lines}</div>`;

  root.innerHTML = head + conclusion + table + transcript;

  // Клик по цитате прокручивает транскрипт к нужному месту. Без подсветки
  // прокрутка бесполезна: страница дёрнулась, а куда смотреть — непонятно.
  root.querySelectorAll('.cite[data-turn]').forEach(el => {
    el.onclick = () => {
      const target = document.getElementById(turnId(el.dataset.turn));
      if (!target) return;
      target.scrollIntoView({behavior: 'smooth', block: 'center'});
      target.classList.remove('hit');
      void target.offsetWidth;              // перезапуск анимации
      target.classList.add('hit');
    };
  });
  return root;
}
