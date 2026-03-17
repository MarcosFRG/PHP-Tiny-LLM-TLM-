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
    foreach ($ids as $id) {
      $tokens[] = $this->vocab[$id] ?? '<UNK>';
    }
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

class PPMTrie {
  private array $root;
  private array $globalCounts;
  private int $totalTokens;

  public function __construct() {
    $this->root = [];
    $this->globalCounts = [];
    $this->totalTokens = 0;
  }

  public function add(array $context, int $nextToken): void {
    $node = &$this->root;
    foreach ($context as $token) {
      if (!isset($node['children'][$token])) {
        $node['children'][$token] = ['counts' => [], 'children' => []];
      }
      $node = &$node['children'][$token];
    }
    if (!isset($node['counts'][$nextToken])) {
      $node['counts'][$nextToken] = 1;
    } else {
      $node['counts'][$nextToken]++;
    }
    if (!isset($this->globalCounts[$nextToken])) {
      $this->globalCounts[$nextToken] = 1;
    } else {
      $this->globalCounts[$nextToken]++;
    }
    $this->totalTokens++;
  }

  private function getNode(array $context): ?array {
    $node = $this->root;
    foreach ($context as $token) {
      if (!isset($node['children'][$token])) {
        return null;
      }
      $node = $node['children'][$token];
    }
    return $node;
  }

  public function predict(array $context, int $maxOrder, float $temperature = 1.0, ?int $topK = null, ?float $topP = null): array {
    $order = min(count($context), $maxOrder);
    $probs = [];
    $weight = 1.0;
    $seen = [];

    for ($o = $order; $o >= 0; $o--) {
      $subContext = array_slice($context, -$o);
      $node = $o === 0 ? ['counts' => $this->globalCounts] : $this->getNode($subContext);
      if (!$node) continue;

      $counts = $node['counts'];
      if (empty($counts)) continue;

      $total = array_sum($counts);
      $numSymbols = count($counts);
      $escapeProb = $numSymbols / ($total + $numSymbols);
      $factor = $weight * (1 - $escapeProb);

      foreach ($counts as $token => $freq) {
        if (!isset($seen[$token])) {
          $probs[$token] = ($probs[$token] ?? 0) + $factor * ($freq / $total);
          $seen[$token] = true;
        }
      }
      $weight *= $escapeProb;
      if ($weight < 1e-6) break;
    }

    if (empty($probs)) {
      $probs[0] = 1.0;
    }

    if ($temperature != 1.0) {
      $sum = 0.0;
      foreach ($probs as $token => $p) {
        $p = pow($p, 1.0 / max($temperature, 0.01));
        $probs[$token] = $p;
        $sum += $p;
      }
      if ($sum > 0) {
        foreach ($probs as $token => $p) {
          $probs[$token] = $p / $sum;
        }
      }
    }

    if ($topK !== null && $topK > 0 && count($probs) > $topK) {
      arsort($probs);
      $probs = array_slice($probs, 0, $topK, true);
      $sum = array_sum($probs);
      if ($sum > 0) {
        foreach ($probs as $token => $p) {
          $probs[$token] = $p / $sum;
        }
      }
    }

    if ($topP !== null && $topP > 0.0 && $topP < 1.0) {
      arsort($probs);
      $cum = 0.0;
      $filtered = [];
      foreach ($probs as $token => $p) {
        $cum += $p;
        $filtered[$token] = $p;
        if ($cum >= $topP) {
          break;
        }
      }
      $sum = array_sum($filtered);
      if ($sum > 0) {
        foreach ($filtered as $token => $p) {
          $filtered[$token] = $p / $sum;
        }
      }
      $probs = $filtered;
    }

    return $probs;
  }

  // Guardado binario para mayor velocidad
  public function save(string $path): void {
    $fp = fopen($path, 'wb');
    if (!$fp) return;

    // Función recursiva para escribir el trie
    $writeNode = function($node) use (&$writeNode, $fp) {
      // Escribir counts
      $counts = $node['counts'] ?? [];
      fwrite($fp, pack('V', count($counts))); // Número de entradas en counts
      foreach ($counts as $token => $count) {
        fwrite($fp, pack('V', $token));
        fwrite($fp, pack('V', $count));
      }
  
      // Escribir children
      $children = $node['children'] ?? [];
      fwrite($fp, pack('V', count($children))); // Número de hijos
      foreach ($children as $token => $child) {
        fwrite($fp, pack('V', $token));
        $writeNode($child);
      }
    };

    // Escribir globalCounts
    fwrite($fp, pack('V', count($this->globalCounts)));
    foreach ($this->globalCounts as $token => $count) {
      fwrite($fp, pack('V', $token));
      fwrite($fp, pack('V', $count));
    }

    // Escribir totalTokens
    fwrite($fp, pack('V', $this->totalTokens));

    // Escribir root
    $writeNode($this->root);

    fclose($fp);
  }

  // Carga binaria
  public function load(string $path): void {
    if (!file_exists($path)) return;

    $fp = fopen($path, 'rb');
    if (!$fp) return;

    // Función recursiva para leer el trie
    $readNode = function() use (&$readNode, $fp) {
      $node = ['counts' => [], 'children' => []];
  
      // Leer counts
      $countSize = unpack('V', fread($fp, 4))[1];
      for ($i = 0; $i < $countSize; $i++) {
        $token = unpack('V', fread($fp, 4))[1];
        $count = unpack('V', fread($fp, 4))[1];
        $node['counts'][$token] = $count;
      }
  
      // Leer children
      $childrenSize = unpack('V', fread($fp, 4))[1];
      for ($i = 0; $i < $childrenSize; $i++) {
        $token = unpack('V', fread($fp, 4))[1];
        $node['children'][$token] = $readNode();
      }
  
      return $node;
    };

    // Leer globalCounts
    $globalSize = unpack('V', fread($fp, 4))[1];
    for ($i = 0; $i < $globalSize; $i++) {
      $token = unpack('V', fread($fp, 4))[1];
      $count = unpack('V', fread($fp, 4))[1];
      $this->globalCounts[$token] = $count;
    }

    // Leer totalTokens
    $this->totalTokens = unpack('V', fread($fp, 4))[1];

    // Leer root
    $this->root = $readNode();

    fclose($fp);
  }
}

class LLM {
  private Tokenizer $tokenizer;
  private PPMTrie $trie;
  private string $modelDir;
  private int $maxContext;
  private array $specialTokens = ['<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|EOS|>', '<EOS>'];
  private array $noSpaceBefore = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '”', '’', '»', '¡', '¿'];
  private array $noSpaceAfter = ['(', '[', '{', '“', '‘', '«'];

  public function __construct(string $modelDir, int $maxContext = 512) {
    $this->modelDir = $modelDir;
    $this->maxContext = $maxContext;
    if (!is_dir($modelDir)) mkdir($modelDir, 0777, true);

    $tokenizerPath = $modelDir . '/tokenizer.json';
    $triePath = $modelDir . '/model.ppm';

    if (file_exists($tokenizerPath)) {
      $this->tokenizer = new Tokenizer();
      $this->tokenizer->load($tokenizerPath);
    } else {
      $this->tokenizer = new Tokenizer();
      $this->tokenizer->encode($this->specialTokens);
    }

    if (file_exists($triePath)) {
      $this->trie = new PPMTrie();
      $this->trie->load($triePath);
    } else {
      $this->trie = new PPMTrie();
    }
  }

  public function train(string $text): void {
    $tokens = $this->tokenizer->tokenize($text, false);
    $ids = $this->tokenizer->encode($tokens);
    $len = count($ids);

    for ($i = 0; $i < $len - 1; $i++) {
      $maxOrder = min($this->maxContext, $i + 1);
      for ($order = 1; $order <= $maxOrder; $order++) {
        $context = array_slice($ids, $i - $order + 1, $order);
        $next = $ids[$i + 1];
        $this->trie->add($context, $next);
      }
    }

    $this->tokenizer->save($this->modelDir . '/tokenizer.json');
    $this->trie->save($this->modelDir . '/model.ppm');
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
    if(!str_contains($prompt, '<|')) $prompt = "<|USER|>\n".trim($prompt)."\n<|EOS|>\n<|ASSISTANT|>\n";

    $allStopTokens = array_merge($stopTokens, ['<|EOS|>', '<EOS>']);

    $tokens = $this->tokenizer->tokenize($prompt, true);
    $ids = $this->tokenizer->encode($tokens);
    if ($this->tokenizer->getVocabSize() === 0) return $prompt;

    $freqCount = [];
    $generatedIds = [];

    for ($i = 0; $i < $maxTokens; $i++) {
      $context = array_slice($ids, -$this->maxContext);
      $probs = $this->trie->predict($context, $this->maxContext, $temperature, $topK, $topP);

      if ($frequencyPenalty > 0) {
        foreach ($probs as $id => $p) {
          $count = $freqCount[$id] ?? 0;
          $probs[$id] = $p / (1 + $frequencyPenalty * $count);
        }
        $sum = array_sum($probs);
        if ($sum > 0) {
          foreach ($probs as $id => $p) {
            $probs[$id] = $p / $sum;
          }
        }
      }

      if ($repetitionPenalty != 1.0) {
        foreach ($probs as $id => $p) {
          if (isset($freqCount[$id]) && $freqCount[$id] > 0) {
            $factor = 1.0 / $repetitionPenalty;
            $probs[$id] = $p * $factor;
          }
        }
        $sum = array_sum($probs);
        if ($sum > 0) {
          foreach ($probs as $id => $p) {
            $probs[$id] = $p / $sum;
          }
        }
      }

      if ($presencePenalty != 0.0) {
        $factor = exp(-$presencePenalty);
        foreach ($probs as $id => $p) {
          if (isset($freqCount[$id]) && $freqCount[$id] > 0) {
            $probs[$id] = $p * $factor;
          }
        }
        $sum = array_sum($probs);
        if ($sum > 0) {
          foreach ($probs as $id => $p) {
            $probs[$id] = $p / $sum;
          }
        }
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
      if ($selected === null) {
        $selected = array_key_first($probs) ?? 0;
      }

      $ids[] = $selected;
      $generatedIds[] = $selected;
      $freqCount[$selected] = ($freqCount[$selected] ?? 0) + 1;

      $token = $this->tokenizer->decode([$selected])[0];
      if (in_array($token, $allStopTokens, true)) {
        break;
      }
    }

    $outputTokens = $this->tokenizer->decode($generatedIds);

    $filteredTokens = array_filter($outputTokens, function($token) {
      return !in_array($token, $this->specialTokens);
    });

    return $this->joinTokens($filteredTokens);
  }

  private function joinTokens(array $tokens): string {
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
        if (!in_array($token, $this->noSpaceBefore) && !in_array($prevToken, $this->noSpaceAfter)) {
          $addSpace = true;
        }
      }
      if ($addSpace) {
        $result .= ' ';
      }
      $result .= $token;
      $prevToken = $token;
    }
    return $result;
  }

  public function getVocabSize(): int {
    return $this->tokenizer->getVocabSize();
  }
}