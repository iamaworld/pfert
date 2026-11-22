import test from 'node:test';
import assert from 'node:assert/strict';
import { similarity, validFeatures, toClassification } from './server.js';

const base = { hash: '0000000000000000', histogram: Array(24).fill(0), grid: Array(64).fill(0.5) };
for (let channel = 0; channel < 3; channel++) base.histogram[channel * 8 + 2] = 1;

test('identical features score 100%', () => assert.equal(similarity(base, base), 1));
test('visually different features score below an identical photo', () => {
  const different = { hash: 'ffffffffffffffff', histogram: Array(24).fill(0), grid: Array(64).fill(0) };
  for (let channel = 0; channel < 3; channel++) different.histogram[channel * 8 + 6] = 1;
  assert.ok(similarity(base, different) < similarity(base, base));
});
test('reject malformed features', () => {
  assert.equal(validFeatures(base), true);
  assert.equal(validFeatures({ ...base, hash: 'invalid' }), false);
  assert.equal(validFeatures({ ...base, grid: Array(63).fill(0) }), false);
});
test('classification supports any number of catalog classes', () => {
  const horse = id => ({ horse: { id, name: id } });
  const one = toClassification([horse('a')], entry => 1);
  assert.equal(one.length, 1);
  assert.equal(one[0].similarity, 1);
  const three = toClassification([horse('a'), horse('b'), horse('c')], entry => entry.horse.id === 'a' ? 2 : 1);
  assert.equal(three.length, 3);
  assert.equal(three[0].horseId, 'a');
  assert.equal(three[0].similarity, 1);
  assert.equal(three[1].similarity, 0.5);
  assert.deepEqual(toClassification([], () => 1), []);
});
