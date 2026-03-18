<?php
class Tokenizer {
  private array $vocab = [];
  private array $reverseVocab = [];
  private int $nextId = 0;
  private array $cache = [];

  public function __construct(array $vocab = []) {
    if (!empty($vocab)) {
      $this->vocab = $vocab;
      $this->reverseVocab = array_flip($vocab);
      $this->nextId = count($vocab);
    }
  }

  public function tokenize(string $text, bool $useCache = true): array {
    $text = trim($text);
    if ($text === '') return [];
    if ($useCache && isset($this->cache[$text])) return $this->cache[$text];
    preg_match_all('/(<\|[^>]+\|>|<[^>]+>|\p{L}+|\p{N}+|\p{So}+|\p{P}+|\n)/u', $text, $matches);
    $tokens = $matches[0];
    if($useCache) $this->cache[$text] = $tokens;
    return $tokens;
  }

  public function encode(array $tokens): array {
    $ids = [];
    foreach ($tokens as $token) {
      if (!isset($this->reverseVocab[$token])) {
        $this->reverseVocab[$token] = $this->nextId;
        $this->vocab[$this->nextId] = $token;
        $ids[] = $this->nextId;
        $this->nextId++;
      } else {
        $ids[] = $this->reverseVocab[$token];
      }
    }
    return $ids;
  }

  public function decode(array $ids): array {
    $tokens = [];
    foreach ($ids as $id) $tokens[] = $this->vocab[$id] ?? '<UNK>';
    return $tokens;
  }

  public function save(string $path): void {
    file_put_contents($path, json_encode([
      'vocab' => $this->vocab,
      'reverseVocab' => $this->reverseVocab,
      'nextId' => $this->nextId
    ]));
  }

  public function load(string $path): void {
    $data = json_decode(file_get_contents($path), true);
    $this->vocab = $data['vocab'];
    $this->reverseVocab = $data['reverseVocab'];
    $this->nextId = $data['nextId'];
  }

  public function getVocabSize(): int {
    return count($this->vocab);
  }
}

class EchoStateLM {
  private int $embedDim = 100;
  private int $reservoirSize = 500;
  private float $lr = 0.1;
  private float $spectralRadius = 0.9;
  private float $sparsity = 0.1;

  private array $embeddings;
  private array $W_in;
  private array $W_res;
  private array $bias;
  private array $W_out;
  private int $vocabSize = 0;

  public function __construct(Tokenizer $tokenizer) {
    $this->initFixedWeights();
    $this->embeddings = [];
    $this->W_out = [];
  }

  private function initFixedWeights(): void {
    // Matriz de entrada dispersa
    $this->W_in = [];
    $totalIn = $this->reservoirSize * $this->embedDim;
    $numNonZeroIn = (int)($totalIn * $this->sparsity);
    for ($k = 0; $k < $numNonZeroIn; $k++) {
      $i = random_int(0, $this->reservoirSize - 1);
      $j = random_int(0, $this->embedDim - 1);
      $val = (mt_rand(-100, 100) / 1000.0);
      $this->W_in[] = [$i, $j, $val];
    }

    // Matriz recurrente dispersa
    $this->W_res = [];
    $totalRes = $this->reservoirSize * $this->reservoirSize;
    $numNonZeroRes = (int)($totalRes * $this->sparsity);
    for ($k = 0; $k < $numNonZeroRes; $k++) {
      $i = random_int(0, $this->reservoirSize - 1);
      $j = random_int(0, $this->reservoirSize - 1);
      $val = (mt_rand(-100, 100) / 1000.0);
      $this->W_res[] = [$i, $j, $val];
    }

    // Escalar para radio espectral aproximado
    $scale = $this->spectralRadius / 1.0;
    foreach ($this->W_res as &$w) $w[2] *= $scale;

    // Bias
    $this->bias = [];
    for ($i = 0; $i < $this->reservoirSize; $i++) $this->bias[] = (mt_rand(-10, 10) / 100.0);
  }

  // Asegura que exista embedding y fila en W_out para un token ID
  private function ensureToken(int $tokenId): void {
    if (!isset($this->embeddings[$tokenId])) {
      // Crear embedding aleatorio
      $vec = [];
      for ($j = 0; $j < $this->embedDim; $j++) $vec[] = (mt_rand(-100, 100) / 1000.0);
      $this->embeddings[$tokenId] = $vec;

      // Crear fila en W_out aleatoria
      $row = [];
      for ($j = 0; $j < $this->reservoirSize; $j++) $row[] = (mt_rand(-10, 10) / 1000.0);
      $this->W_out[$tokenId] = $row;

      // Actualizar vocabSize si es necesario
      if ($tokenId >= $this->vocabSize) $this->vocabSize = $tokenId + 1;
    }
  }

  private function sparseMul(array $spMat, array $vec, int $rows): array {
    $res = array_fill(0, $rows, 0.0);
    foreach ($spMat as $entry) {
      [$i, $j, $val] = $entry;
      $res[$i] += $val * $vec[$j];
    }
    return $res;
  }

  private function updateState(array $state, array $x): array {
    $resPart = $this->sparseMul($this->W_res, $state, $this->reservoirSize);
    $inPart = $this->sparseMul($this->W_in, $x, $this->reservoirSize);
    $newState = [];
    for ($i = 0; $i < $this->reservoirSize; $i++) {
      $val = $resPart[$i] + $inPart[$i] + $this->bias[$i];
      $newState[] = tanh($val);
    }
    return $newState;
  }

  public function trainOnSequence(array $ids): float {
    $len = count($ids);
    if ($len < 2) return 0.0;

    $state = array_fill(0, $this->reservoirSize, 0.0);
    $totalLoss = 0.0;

    for ($t = 0; $t < $len - 1; $t++) {
      $tokenId = $ids[$t];
      $nextId = $ids[$t + 1];

      // Asegurar que ambos tokens existen
      $this->ensureToken($tokenId);
      $this->ensureToken($nextId);

      $x = $this->embeddings[$tokenId];
      $state = $this->updateState($state, $x);

      // Calcular logits (solo para los tokens que existen en W_out)
      $logits = [];
      foreach ($this->W_out as $id => $row) {
        $dot = 0.0;
        for ($j = 0; $j < $this->reservoirSize; $j++) $dot += $row[$j] * $state[$j];
        $logits[$id] = $dot;
      }

      // Softmax y pérdida
      $maxLogit = max($logits);
      $expSum = 0.0;
      $probs = [];
      foreach ($logits as $id => $l) {
        $e = exp($l - $maxLogit);
        $probs[$id] = $e;
        $expSum += $e;
      }
      $loss = -log(($probs[$nextId] ?? 0.0) / $expSum + 1e-10);
      $totalLoss += $loss;

      // Gradiente de la pérdida con respecto a los logits
      $dLogits = [];
      foreach ($logits as $id => $l) $dLogits[$id] = ($probs[$id] / $expSum) - ($id == $nextId ? 1.0 : 0.0);

      foreach ($dLogits as $id => $d) {
        if (abs($d) < 1e-8) continue;
        $row = &$this->W_out[$id];
        for ($j = 0; $j < $this->reservoirSize; $j++) $row[$j] -= $this->lr * $d * $state[$j];
      }
    }

    return $totalLoss / ($len - 1);
  }

  public function generate(array $contextIds, int $maxTokens, float $temperature = 1.0, ?int $topK = null, ?float $topP = null, float $repetitionPenalty = 1.0, float $presencePenalty = 0.0, float $frequencyPenalty = 0.0): array {
    $state = array_fill(0, $this->reservoirSize, 0.0);
    foreach ($contextIds as $tokenId) {
      $this->ensureToken($tokenId);
      $x = $this->embeddings[$tokenId];
      $state = $this->updateState($state, $x);
    }

    $generated = [];
    $freqCount = [];

    for ($t = 0; $t < $maxTokens; $t++) {
      $logits = [];
      foreach ($this->W_out as $id => $row) {
        $dot = 0.0;
        for ($j = 0; $j < $this->reservoirSize; $j++) $dot += $row[$j] * $state[$j];
        $logits[$id] = $dot;
      }

      if (empty($logits)) break;

      $maxLogit = max($logits);
      $expSum = 0.0;
      $probs = [];
      foreach ($logits as $id => $l) {
        $e = exp(($l - $maxLogit) / max($temperature, 0.01));
        $probs[$id] = $e;
        $expSum += $e;
      }
      foreach ($probs as $id => $e) $probs[$id] = $e / $expSum;

      if ($topK !== null && $topK > 0 && count($probs) > $topK) {
        arsort($probs);
        $probs = array_slice($probs, 0, $topK, true);
        $sum = array_sum($probs);
        foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
      }

      if ($topP !== null && $topP > 0.0 && $topP < 1.0) {
        arsort($probs);
        $cum = 0.0;
        $filtered = [];
        foreach ($probs as $id => $p) {
          $cum += $p;
          $filtered[$id] = $p;
          if ($cum >= $topP) break;
        }
        $sum = array_sum($filtered);
        foreach ($filtered as $id => $p) $filtered[$id] = $p / $sum;
        $probs = $filtered;
      }

      if ($frequencyPenalty > 0) {
        foreach ($probs as $id => $p) {
          $count = $freqCount[$id] ?? 0;
          $probs[$id] = $p / (1 + $frequencyPenalty * $count);
        }
        $sum = array_sum($probs);
        foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
      }

      if ($repetitionPenalty != 1.0) {
        foreach ($probs as $id => $p) {
          if (isset($freqCount[$id])) $probs[$id] = $p / $repetitionPenalty;
        }
        $sum = array_sum($probs);
        foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
      }

      if ($presencePenalty != 0.0) {
        $factor = exp(-$presencePenalty);
        foreach ($probs as $id => $p) {
          if (isset($freqCount[$id])) $probs[$id] = $p * $factor;
        }
        $sum = array_sum($probs);
        foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
      }

      $rand = mt_rand() / mt_getrandmax();
      $cum = 0.0;
      $selected = null;
      foreach ($probs as $id => $p) {
        $cum += $p;
        if ($rand <= $cum) {
          $selected = $id;
          break;
        }
      }
      if ($selected === null) $selected = array_key_first($probs) ?? 0;

      $generated[] = $selected;
      $freqCount[$selected] = ($freqCount[$selected] ?? 0) + 1;

      $this->ensureToken($selected);
      $x = $this->embeddings[$selected];
      $state = $this->updateState($state, $x);
    }

    return $generated;
  }

  public function save(string $path): void {
    $fp = fopen($path, 'wb');
    if (!$fp) return;
    fwrite($fp, pack('V', $this->embedDim));
    fwrite($fp, pack('V', $this->reservoirSize));
    fwrite($fp, pack('V', $this->vocabSize));

    // Guardar embeddings (solo los que existen)
    fwrite($fp, pack('V', count($this->embeddings)));
    foreach ($this->embeddings as $id => $vec) {
      fwrite($fp, pack('V', $id));
      for ($j = 0; $j < $this->embedDim; $j++) fwrite($fp, pack('f', $vec[$j]));
    }

    // Guardar W_in
    fwrite($fp, pack('V', count($this->W_in)));
    foreach ($this->W_in as $e) {
      fwrite($fp, pack('V', $e[0]));
      fwrite($fp, pack('V', $e[1]));
      fwrite($fp, pack('f', $e[2]));
    }

    // Guardar W_res
    fwrite($fp, pack('V', count($this->W_res)));
    foreach ($this->W_res as $e) {
      fwrite($fp, pack('V', $e[0]));
      fwrite($fp, pack('V', $e[1]));
      fwrite($fp, pack('f', $e[2]));
    }

    // Guardar bias
    for ($i = 0; $i < $this->reservoirSize; $i++) fwrite($fp, pack('f', $this->bias[$i]));

    // Guardar W_out
    fwrite($fp, pack('V', count($this->W_out)));
    foreach ($this->W_out as $id => $row) {
      fwrite($fp, pack('V', $id));
      for ($j = 0; $j < $this->reservoirSize; $j++) fwrite($fp, pack('f', $row[$j]));
    }
    fclose($fp);
  }

  public function load(string $path): void {
    $fp = fopen($path, 'rb');
    if (!$fp) return;
    $embedDim = unpack('V', fread($fp, 4))[1];
    $reservoirSize = unpack('V', fread($fp, 4))[1];
    $vocabSize = unpack('V', fread($fp, 4))[1];
    $this->embedDim = $embedDim;
    $this->reservoirSize = $reservoirSize;
    $this->vocabSize = $vocabSize;

    // Cargar embeddings
    $numEmb = unpack('V', fread($fp, 4))[1];
    $this->embeddings = [];
    for ($k = 0; $k < $numEmb; $k++) {
      $id = unpack('V', fread($fp, 4))[1];
      $vec = [];
      for ($j = 0; $j < $embedDim; $j++) $vec[] = unpack('f', fread($fp, 4))[1];
      $this->embeddings[$id] = $vec;
    }

    // Cargar W_in
    $numIn = unpack('V', fread($fp, 4))[1];
    $this->W_in = [];
    for ($k = 0; $k < $numIn; $k++) {
      $i = unpack('V', fread($fp, 4))[1];
      $j = unpack('V', fread($fp, 4))[1];
      $val = unpack('f', fread($fp, 4))[1];
      $this->W_in[] = [$i, $j, $val];
    }

    // Cargar W_res
    $numRes = unpack('V', fread($fp, 4))[1];
    $this->W_res = [];
    for ($k = 0; $k < $numRes; $k++) {
      $i = unpack('V', fread($fp, 4))[1];
      $j = unpack('V', fread($fp, 4))[1];
      $val = unpack('f', fread($fp, 4))[1];
      $this->W_res[] = [$i, $j, $val];
    }

    // Cargar bias
    $this->bias = [];
    for ($i = 0; $i < $reservoirSize; $i++) $this->bias[] = unpack('f', fread($fp, 4))[1];

    // Cargar W_out
    $numOut = unpack('V', fread($fp, 4))[1];
    $this->W_out = [];
    for ($k = 0; $k < $numOut; $k++) {
      $id = unpack('V', fread($fp, 4))[1];
      $row = [];
      for ($j = 0; $j < $reservoirSize; $j++) $row[] = unpack('f', fread($fp, 4))[1];
      $this->W_out[$id] = $row;
    }
    fclose($fp);
  }
}

class LLM {
  private Tokenizer $tokenizer;
  private ?EchoStateLM $model;
  private string $modelDir;
  private int $maxContext;

  public function __construct(string $modelDir, int $maxContext = 512) {
    $this->modelDir = $modelDir;
    $this->maxContext = $maxContext;
    if (!is_dir($modelDir)) mkdir($modelDir, 0777, true);

    $tokenizerPath = $modelDir . '/tokenizer.json';
    $modelPath = $modelDir . '/model.bin';

    if (file_exists($tokenizerPath)) {
      $this->tokenizer = new Tokenizer();
      $this->tokenizer->load($tokenizerPath);
    } else {
      $this->tokenizer = new Tokenizer();
      $this->tokenizer->encode(['<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|EOS|>']);
    }

    if (file_exists($modelPath)) {
      $this->model = new EchoStateLM($this->tokenizer);
      $this->model->load($modelPath);
    } else {
      $this->model = new EchoStateLM($this->tokenizer);
    }
  }

  public function train(string $text): void {
    $batches = preg_split('/(?<=<\|EOS\|>)\n\s*\n\s*(?=<\|)/', $text);
    foreach ($batches as $batch) {
      $batch = trim($batch);
      if (empty($batch)) continue;
      if (!str_ends_with($batch, '<|EOS|>')) $batch .= '<|EOS|>';

      $tokens = $this->tokenizer->tokenize($batch, false);
      $ids = $this->tokenizer->encode($tokens);
      if (count($ids) < 2) continue;

      // Asegurar que el modelo conozca todos los IDs (ya se hace internamente)
      $loss = $this->model->trainOnSequence($ids);
    }

    $this->save();
  }

  public function generate(
    string $prompt,
    int $maxTokens = 50,
    float $temperature = 0.8,
    ?int $topK = null,
    float $frequencyPenalty = 0.0,
    array $stopTokens = [],
    ?float $topP = null,
    float $repetitionPenalty = 1.0,
    float $presencePenalty = 0.0
  ): string {
    if (!$this->tokenizer->getVocabSize()) return $prompt;

    if (!str_contains($prompt, '<|')) $prompt = "<|USER|>\n" . trim($prompt) . "\n<|EOS|>\n<|ASSISTANT|>\n";

    $tokens = $this->tokenizer->tokenize($prompt, true);
    $contextIds = $this->tokenizer->encode($tokens);
    if (count($contextIds) > $this->maxContext) $contextIds = array_slice($contextIds, -$this->maxContext);

    $generatedIds = $this->model->generate($contextIds, $maxTokens, $temperature, $topK, $topP, $repetitionPenalty, $presencePenalty, $frequencyPenalty);

    $outputTokens = $this->tokenizer->decode($generatedIds);
    $filtered = array_filter($outputTokens, fn($t) => !in_array($t, ['<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|EOS|>']));
    return $this->joinTokens($filtered);
  }

  private function joinTokens(array $tokens): string {
    $noSpaceBefore = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '”', '’', '»'];
    $noSpaceAfter = ['(', '[', '{', '“', '‘', '«', '¡', '¿'];
    $result = '';
    $prevToken = '';
    foreach ($tokens as $token) {
      if ($token === "\n") {
        $result .= "\n";
        $prevToken = "\n";
        continue;
      }
      $addSpace = false;
      if ($result !== '') {
        if (!in_array($token, $noSpaceBefore) && !in_array($prevToken, $noSpaceAfter)) $addSpace = true;
      }
      if ($addSpace) $result .= ' ';
      $result .= $token;
      $prevToken = $token;
    }
    return $result;
  }

  private function save(): void {
    $this->tokenizer->save($this->modelDir . '/tokenizer.json');
    if ($this->model) $this->model->save($this->modelDir . '/model.bin');
  }

  public function getVocabSize(): int {
    return $this->tokenizer->getVocabSize();
  }

  public function deleteModelFiles(): void {
    $files = [
      $this->modelDir . '/tokenizer.json',
      $this->modelDir . '/model.bin',
    ];
    foreach ($files as $file) {
      if (file_exists($file)) unlink($file);
    }
    // Reiniciar modelo
    $this->model = new EchoStateLM($this->tokenizer);
  }
}