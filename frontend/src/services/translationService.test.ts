import { describe, expect, it } from 'vitest';

import { glossKeyVariants } from './translationService';

describe('translationService gloss keys', () => {
  it('returns lowercase, snake_case, and compact variants', () => {
    expect(glossKeyVariants('Thank You')).toContain('thank you');
    expect(glossKeyVariants('Thank You')).toContain('thank_you');
    expect(glossKeyVariants('hello world')).toContain('helloworld');
  });
});
