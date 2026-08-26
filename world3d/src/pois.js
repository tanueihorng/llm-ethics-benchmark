// Landmark definitions for miniature Singapore. Positions are absolute
// (Marina Bay water at the origin); `pos` is the fly-to/hit-sphere centre.

export const POIS = [
  {
    id: 'mbs',
    name: 'Marina Bay Sands',
    sub: 'three towers, one sky park',
    body: 'Three glass towers lean into the evening and a boat of light rests on ' +
          'their shoulders. When dark falls the bay turns theatre and the sky park ' +
          'sweeps its beams across the clouds.',
    pos: [35, 95, 235], viewDist: 190, viewElev: 75,
    hits: [[-25, 95, 235, 32], [35, 95, 235, 32], [95, 95, 235, 32], [35, 192, 235, 58]],
  },
  {
    id: 'flyer',
    name: 'Singapore Flyer',
    sub: 'the giant wheel',
    body: 'A hundred and sixty-five metres of slow, steady turning. From the top ' +
          'you can see the whole bay hold its breath — and from the ground, you can ' +
          'watch the wheel paint its rim in light.',
    pos: [250, 98, -55], viewDist: 185, viewElev: 70,
    hits: [[250, 96, -55, 50]],
  },
  {
    id: 'merlion',
    name: 'The Merlion',
    sub: 'keeper of the spring',
    body: 'Half lion, half fish, entirely unbothered by weather. It pours the same ' +
          'bright arc of water into the bay it has poured for fifty years — a small ' +
          'river running constant as clockwork.',
    pos: [-95, 12, 95], viewDist: 80, viewElev: 32,
    hits: [[-95, 12, 95, 15]],
  },
  {
    id: 'gardens',
    name: 'Gardens by the Bay',
    sub: 'where the supertrees glow',
    body: 'Steel trees taller than houses, breathing light instead of wind. After ' +
          'sundown they hum a garden rhapsody in green and gold, and the great glass ' +
          'clouds of the conservatories sleep beside them.',
    pos: [230, 30, 190], viewDist: 150, viewElev: 80,
    hits: [[230, 30, 190, 44]],
  },
  {
    id: 'museum',
    name: 'ArtScience Museum',
    sub: 'the lotus of the bay',
    body: 'A white lotus with ten petals, open where art meets science on the ' +
          'water\'s edge. It has no doors in this miniature — only the question it ' +
          'was built to ask.',
    pos: [10, 14, 150], viewDist: 100, viewElev: 40,
    hits: [[10, 14, 150, 20]],
  },
];
