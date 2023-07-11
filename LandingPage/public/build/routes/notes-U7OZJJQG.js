import {
  Form
} from "/build/_shared/chunk-2H4SZ7M7.js";
import {
  createHotContext
} from "/build/_shared/chunk-H2LASQMJ.js";
import {
  require_jsx_runtime
} from "/build/_shared/chunk-NMZL6IDN.js";
import "/build/_shared/chunk-BOXFZXVX.js";
import "/build/_shared/chunk-UWV35TSL.js";
import {
  __toESM
} from "/build/_shared/chunk-PNG5AS42.js";

// app/routes/notes.jsx
var import_jsx_runtime2 = __toESM(require_jsx_runtime());

// app/Components/NewNote.jsx
var import_jsx_runtime = __toESM(require_jsx_runtime());
if (!window.$RefreshReg$ || !window.$RefreshSig$ || !window.$RefreshRuntime$) {
  console.warn("remix:hmr: React Fast Refresh only works when the Remix compiler is running in development mode.");
} else {
  prevRefreshReg = window.$RefreshReg$;
  prevRefreshSig = window.$RefreshSig$;
  window.$RefreshReg$ = (type, id) => {
    window.$RefreshRuntime$.register(type, '"app\\\\Components\\\\NewNote.jsx"' + id);
  };
  window.$RefreshSig$ = window.$RefreshRuntime$.createSignatureFunctionForTransform;
}
var prevRefreshReg;
var prevRefreshSig;
if (import.meta) {
  import.meta.hot = createHotContext(
    //@ts-expect-error
    "app\\Components\\NewNote.jsx"
  );
  import.meta.hot.lastModified = "1688728969089.1558";
}
function NewNote() {
  return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
    className: " create_notes",
    children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Form, {
      method: "Post",
      children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
        type: "text",
        name: "name",
        placeholder: "enter patient name"
      }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("br", {}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("textarea", {
        name: "notes",
        placeholder: "enter patient notes"
      }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("br", {}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
        children: "create notes"
      })]
    })
  });
}
_c = NewNote;
var _c;
$RefreshReg$(_c, "NewNote");
window.$RefreshReg$ = prevRefreshReg;
window.$RefreshSig$ = prevRefreshSig;

// app/routes/notes.jsx
if (!window.$RefreshReg$ || !window.$RefreshSig$ || !window.$RefreshRuntime$) {
  console.warn("remix:hmr: React Fast Refresh only works when the Remix compiler is running in development mode.");
} else {
  prevRefreshReg = window.$RefreshReg$;
  prevRefreshSig = window.$RefreshSig$;
  window.$RefreshReg$ = (type, id) => {
    window.$RefreshRuntime$.register(type, '"app\\\\routes\\\\notes.jsx"' + id);
  };
  window.$RefreshSig$ = window.$RefreshRuntime$.createSignatureFunctionForTransform;
}
var prevRefreshReg;
var prevRefreshSig;
if (import.meta) {
  import.meta.hot = createHotContext(
    //@ts-expect-error
    "app\\routes\\notes.jsx"
  );
  import.meta.hot.lastModified = "1688727244928.9153";
}
function About() {
  return /* @__PURE__ */ (0, import_jsx_runtime2.jsx)("div", {
    className: "notes_div",
    children: /* @__PURE__ */ (0, import_jsx_runtime2.jsx)(NewNote, {})
  });
}
_c2 = About;
var _c2;
$RefreshReg$(_c2, "About");
window.$RefreshReg$ = prevRefreshReg;
window.$RefreshSig$ = prevRefreshSig;
export {
  About as default
};
//# sourceMappingURL=/build/routes/notes-U7OZJJQG.js.map
