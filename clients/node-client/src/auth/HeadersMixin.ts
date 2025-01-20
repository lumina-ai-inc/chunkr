export class HeadersMixin {
  protected _apiKey!: string;

  protected getApiKey(): string {
    if (!this._apiKey) {
      throw new Error("API key not set");
    }
    return this._apiKey;
  }

  protected headers(): Record<string, string> {
    return {
      Authorization: this.getApiKey(),
    };
  }
}
