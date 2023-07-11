import {
  require_client
} from "/build/_shared/chunk-ZWGWGGVF.js";
import "/build/_shared/chunk-GIAAE3CH.js";
import {
  RemixBrowser
} from "/build/_shared/chunk-2H4SZ7M7.js";
import {
  createHotContext
} from "/build/_shared/chunk-H2LASQMJ.js";
import {
  require_jsx_runtime
} from "/build/_shared/chunk-NMZL6IDN.js";
import {
  require_react
} from "/build/_shared/chunk-BOXFZXVX.js";
import "/build/_shared/chunk-UWV35TSL.js";
import {
  __toESM
} from "/build/_shared/chunk-PNG5AS42.js";

// app/entry.client.jsx
var import_jsx_runtime = __toESM(require_jsx_runtime());
var import_react2 = __toESM(require_react());
var import_client = __toESM(require_client());
if (import.meta) {
  import.meta.hot = createHotContext(
    //@ts-expect-error
    "app\\entry.client.jsx"
  );
  import.meta.hot.lastModified = "1688008292895.9695";
}
(0, import_react2.startTransition)(() => {
  (0, import_client.hydrateRoot)(
    document,
    /* @__PURE__ */ (0, import_jsx_runtime.jsx)(import_react2.StrictMode, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(RemixBrowser, {}) })
  );
});
//# sourceMappingURL=/build/entry.client-ACFSEPTX.js.map
