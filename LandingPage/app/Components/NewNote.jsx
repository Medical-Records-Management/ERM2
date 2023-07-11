import { Form } from "@remix-run/react"

export default function NewNote()
{
    return(
        <div className=" create_notes">
            <Form method="Post">
                <input type="text" name="name" placeholder="enter patient name"/><br />
                <textarea name="notes" placeholder="enter patient notes"></textarea><br />
                <button>create notes</button>
            </Form>
            
        </div>
    )
}