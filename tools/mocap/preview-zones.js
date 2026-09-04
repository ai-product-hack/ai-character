// Предпросмотр авторинга шире боевого слоя эмоции: он показывает направление
// глаз, чтобы оператор видел качество захвата. В сохранённом клипе данные
// остаются сырыми, а рантайм по-прежнему отфильтровывает eyeLook* и отдаёт
// взгляд только Microbehavior.

import { LAYERS, RULES } from '../../avatar/src/zones.js';

export function previewLayerFor(name) {
  const rule = RULES.get(name);
  if (rule?.layers.has(LAYERS.EMOTION)) return LAYERS.EMOTION;
  if (name.startsWith('eyeLook') && rule?.layers.has(LAYERS.IDLE)) return LAYERS.IDLE;
  return null;
}

