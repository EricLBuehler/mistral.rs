# CustomWorld Pattern Reference

The `CustomWorld` class in `tests/support/world.ts` is the state container for each Cucumber scenario.

## Key Rules

- Every step definition must use `async function (this: CustomWorld)` (no arrow functions)
- Fields are initialized in `Before` hooks per tag (`@api`, `@ui`, `@agent`)
- `testData` is a generic `Record<string, any>` for arbitrary scenario state
- `baseUrl` defaults to `http://localhost:3000` (override via `BASE_URL` env var)

## Field Reference

```typescript
export interface TestWorld extends World {
  // API testing
  apiContext: APIRequestContext;  // Playwright HTTP client
  response: any;                 // Last HTTP response object
  responseBody: any;             // Parsed JSON body
  authToken: string;             // JWT token

  // UI testing
  browser: Browser;              // Chromium instance
  context: BrowserContext;       // Context with video recording
  page: Page;                    // Active page

  // Agent testing
  agentResponse: any;            // Agent chat response
  agentToolCalls: any[];         // Tool invocations
  agentStreamChunks: string[];   // SSE chunks

  // Video
  videoPath: string | null;      // Path to recorded WebM
  tracePath: string | null;      // Path to trace ZIP

  // Shared
  baseUrl: string;
  testData: Record<string, any>;
}
```

## Usage in Steps

```typescript
import { Given } from '@cucumber/cucumber';
import { CustomWorld } from '../support/world';

Given('I am authenticated', async function (this: CustomWorld) {
  const res = await this.apiContext.post('/api/auth/login', {
    data: { email: 'test@example.com', password: 'pass' }
  });
  const body = await res.json();
  this.authToken = body.token;
});
```
