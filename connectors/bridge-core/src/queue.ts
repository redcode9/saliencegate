export class SerialSessionQueue {
  readonly #tails = new Map<string, Promise<void>>();

  run<T>(sessionID: string, operation: () => Promise<T> | T): Promise<T> {
    const prior = this.#tails.get(sessionID) ?? Promise.resolve();
    const result = prior.catch(() => undefined).then(operation);
    const tail = result.then(
      () => undefined,
      () => undefined,
    );
    this.#tails.set(sessionID, tail);
    void tail.finally(() => {
      if (this.#tails.get(sessionID) === tail) this.#tails.delete(sessionID);
    });
    return result;
  }

  async drain(): Promise<void> {
    await Promise.all([...this.#tails.values()]);
  }
}
